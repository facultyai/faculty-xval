import json
import logging
import os

import click
import numpy as np

from keras import backend as K
from keras.models import load_model as keras_load
from sklearn.base import clone as sklearn_clone
from sklearn.externals import joblib

from faculty_xval.utilities import keras_clone_and_compile

LOGGER = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)


def load_model(path, model_type):
    """
    Load the model using the method appropriate for its type ("keras" or other).
    
    Parameters
    ----------
    
    path: String 
        File path to look for the model.
    
    model_type: String
        String specifying the type of model to be loaded. Anything other than 
        "keras" will be loaded using joblib.
    
    """
    if model_type == "keras":
        # Load Keras model.
        LOGGER.info("Loading Keras model")
        model = keras_load(path)
        LOGGER.info("Model loading complete")
    else:
        # Load model of other type.
        LOGGER.info("Loading model with joblib")
        model = joblib.load(path)
        LOGGER.info("Model loading complete")
    return model


def clone_model(model, model_type):
    """
    Clone the model using the method appropriate for its type ("keras",
    "sklearn" or other). Reset the state of the model so that each train/test
    split is independent.
    
    Parameters
    ---------- 
    
    model: Scikit-Learn/Keras Model 
        Model to be cloned.
    
    model_type: String
        String specifying the type of model to be cloned. Recognised options
        are "keras" and "sklearn". Any other option results in the function
        returning the input model, thus doing nothing.
    
    Returns
    -------
    
    cloned: Scikit-Learn/Keras Model
        The cloned model with reset state.
    
    """
    if model_type == "keras":
        cloned = keras_clone_and_compile(model)
    elif model_type == "sklearn":
        cloned = sklearn_clone(model)
    else:
        cloned = model
        LOGGER.warning(
            "Model type not recognised. "
            + "Cannot reset the state of the model automatically"
        )
    return cloned


def validate(
    model, features, targets, i_train, i_test, fit_kwargs=None, predict_kwargs=None
):
    """
    Fit the model on specific training data, and predict on specific test data.
    
    Parameters
    ----------
    
    model: sklearn/keras Model
        Model to cross-validate.
    
    features: list of np.array
        Features for training/testing. For multi-input models, the list contains
        multiple Numpy arrays.
    
    targets: list of np.array
        Targets for training/testing. For multi-output models, the list contains
        multiple Numpy arrays. 
    
    i_train: np.array 
        np.array of indices corresponding to the rows used for training
        
    i_test: np.array
        np.array of indices corresponding to the rows used for testing
    
    fit_kwargs: dict, optional, default = None 
        Dictionary of any additional kwargs to be used by the model during 
        fitting.

    predict_kwargs: dict, optional, default = None
        Dictionary of any additional kwargs to be used by the model during 
        prediction.
    
    Returns
    --------
    
    predictions: np.array
        Model predictions.

    """
    if fit_kwargs is None:
        fit_kwargs = {}
    if predict_kwargs is None:
        predict_kwargs = {}

    LOGGER.info("Training the model")
    features_train = [x[i_train] for x in features]
    targets_train = [y[i_train] for y in targets]
    if len(features_train) == 1:
        features_train = features_train[0].copy()
    if len(targets_train) == 1:
        targets_train = targets_train[0].copy()

    model.fit(features_train, targets_train, **fit_kwargs)

    LOGGER.info("Generating model predictions")
    features_test = [x[i_test] for x in features]
    if len(features_test) == 1:
        features_test = features_test[0].copy()

    predictions = model.predict(features_test, **predict_kwargs)
    return np.array(predictions)


@click.command()
@click.argument("input_paths")
def main(input_paths):
    """
    Validate the model for the different train/test splits corresponding to the
    input file paths.
    
    Parameters
    ----------
    
    input_paths: String
        String that defines the paths to load job instructions from. Distinct
        paths are separated by a colon ":".
    
    """
    # Get a list of input file paths.
    input_paths = [x.strip() for x in input_paths.split(":")]

    # Load data.
    LOGGER.info("Loading features and targets from disk")
    with open(input_paths[0], "r") as f:
        _instructions = json.load(f)
    with open(_instructions["features_path"], "r") as f:
        features = json.load(f)
    with open(_instructions["targets_path"], "r") as f:
        targets = json.load(f)

    # Convert datasets to Numpy arrays.
    features = [np.array(x) for x in features]
    targets = [np.array(y) for y in targets]

    # Iterate over train/test splits.
    K.clear_session()
    for input_path in input_paths:
        with open(input_path, "r") as f:
            instructions = json.load(f)
        LOGGER.info("Processing split {}".format(instructions["split_id"]))

        # Load model.
        archetype = load_model(instructions["model_path"], instructions["model_type"])

        # Reset the state of the model to ensure
        # that all splits are independent.
        LOGGER.info("Cloning the model. Resetting the state of the model")
        model = clone_model(archetype, instructions["model_type"])

        # Run validation on specific training and testing datasets.
        predictions = validate(
            model,
            features,
            targets,
            instructions["training_indices"],
            instructions["test_indices"],
            fit_kwargs=instructions["fit_kwargs"],
            predict_kwargs=instructions["predict_kwargs"],
        )

        # Save the predictions alongside an identifier.
        output_dir = os.path.dirname(input_path)
        output_path_predictions = os.path.join(output_dir, "output.json")
        LOGGER.info("Saving predictions to {}".format(output_path_predictions))
        with open(output_path_predictions, "w") as f:
            json.dump({instructions["split_id"]: predictions.tolist()}, f)

        # Clear session to avoid memory build-up.
        K.clear_session()
        del model
        del archetype


if __name__ == "__main__":
    main()
