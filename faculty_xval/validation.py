import os
import json
import logging

import numpy as np

from keras.models import Model
from faculty import client
from sklearn.base import BaseEstimator
from sklearn.externals import joblib

from faculty_xval.utilities import utc_datetime_now
from faculty_xval.utilities import utc_timestamp_now


LOGGER = logging.getLogger(__name__)


class JobsCrossValidatorEncoder(json.JSONEncoder):
    """
    Facilitates the encoding of JobsCrossValidator objects as JSON dictionaries.
    """

    def default(self, o):
        return o.__dict__


def get_chunks(input_list, num_chunks):
    """
    Split a supplied array of train/test indices into smaller chunks, and
    return those chunks as a list of lists.
    """

    chunked = np.array_split(input_list, num_chunks)
    return [x.tolist() for x in chunked]


class JobsCrossValidator(object):
    """
    Use faculty jobs to distribute cross-validation computations across
    multiple servers, and then collate the results.
    """

    def __init__(self, job_id, reference_dir, project_id=None):
        """
        Initialisation function.
        
        Parameters
        ----------
        
        job_id: uuid
            Identifier corresponding to the faculty job which will perform the
            cross-vaildation task. See README.md for instructions on how to
            setup this job.
            
        reference_dir: String
            Path of the directory in which to save results.
        
        project_id: uuid, optional, default = None
            Identifier corresponding to a project on faculty platform. Default
            behaviour is to automatically detect the uuid of the current
            project.
    
        """

        # Get default project ID.
        if project_id is None:
            project_id = os.environ["FACULTY_PROJECT_ID"]

        # Define names of directories used to transfer
        # data to and from individual runs.
        sub_dir = os.path.join(reference_dir, "xval_{}")
        subsub_dir = os.path.join(sub_dir, "split_{}")

        self.job_id = job_id
        self.reference_dir = reference_dir
        self.sub_dir = sub_dir
        self.dump_time = ""
        self.dump_time_format = ""
        self.subsub_dir = subsub_dir
        self.split_ids = []
        self.in_base = "input.json"
        self.out_base = "output.json"
        self.features_base = "features.json"
        self.targets_base = "targets.json"
        self.model_base = None
        self.model_type = None
        self.project_id = project_id
        self.validator_base = "validator.json"

    def _dump(
        self,
        model,
        features,
        targets,
        split_generator,
        fit_kwargs=None,
        predict_kwargs=None,
        dump_time_format="datetime",
    ):
        """
        Writes required information (model, features, targets, split indices,
        and any kwargs) to disk so that it can later be read by multiple
        compute resources during the cross-validation task.
        
        Parameters
        ----------
        
        model: sklearn/keras Model 
            Model to be cross-validated.
            
        features: np.array/pd.DataFrame 
            Feature matrix for training/testing.
            
        targets: np.array/pd.DataFrame 
            Target vector/matrix for training/testing.

        split_generator: generator
            Generator that yields the required train/test indices upon 
            iteration. It must comply with Scikit-Learn's standard for cross
            validation iterators such as `sklearn.model_selection.ShuffleSplit`.
        
        fit_kwargs: dict, optional, default = None 
            Dictionary of any additional kwargs to be used by the model during 
            fitting.

        predict_kwargs: dict, optional, default = None
            Dictionary of any additional kwargs to be used by the model during 
            prediction.
        
        dump_time_format: String, optional, default = "datetime"
            String that controls what format to use when naming the subdirectory 
            which stores the data written to disk. Valid options are "datetime"
            or "timestamp".
        
        Returns
        -------
        
        None, but writes information to disk.
        
        """

        if fit_kwargs is None:
            fit_kwargs = {}
        if predict_kwargs is None:
            predict_kwargs = {}

        # Detect model type.
        if isinstance(model, Model):
            self.model_type = "keras"
        elif isinstance(model, BaseEstimator):
            self.model_type = "sklearn"
        else:
            LOGGER.warning("Model type not recognised")
            self.model_type = "unrecognised"

        # Create subdirectory using UTC time.
        if dump_time_format == "datetime":
            self.dump_time = utc_datetime_now()
        elif dump_time_format == "timestamp":
            self.dump_time = utc_timestamp_now()
        else:
            raise ValueError(
                "`dump_time_format` must be either 'datetime' or 'timestamp'"
            )
        LOGGER.info("Creating run directory " + self.sub_dir.format(self.dump_time))
        os.mkdir(self.sub_dir.format(self.dump_time))
        self.dump_time_format = dump_time_format

        LOGGER.info("Writing model to disk")
        if self.model_type == "keras":
            # Save Keras model.
            self.model_base = "model.h5"
            model.save(
                os.path.join(self.sub_dir.format(self.dump_time), self.model_base)
            )
        else:
            # Save model of other type.
            self.model_base = "model.pkl"
            joblib.dump(
                model,
                os.path.join(self.sub_dir.format(self.dump_time), self.model_base),
            )

        # Convert arrays to lists for storage in JSON.
        LOGGER.info("Writing features and targets to disk")
        if not isinstance(features, list):
            raise TypeError("`features` must be a list of Numpy arrays.")
        if not isinstance(targets, list):
            raise TypeError("`targets` must be a list of Numpy arrays.")
        features = [x.tolist() for x in features]
        targets = [y.tolist() for y in targets]

        # Save features and targets.
        with open(
            os.path.join(self.sub_dir.format(self.dump_time), self.features_base), "w"
        ) as f:
            json.dump(features, f)
        with open(
            os.path.join(self.sub_dir.format(self.dump_time), self.targets_base), "w"
        ) as f:
            json.dump(targets, f)

        # Save instructions for the individual runs.
        for split_id, (i_train, i_test) in enumerate(split_generator):

            # Put together run instructions.
            LOGGER.info("Writing run instructions to disk (split {})".format(split_id))
            instructions = {
                "split_id": split_id,
                "model_path": os.path.join(
                    self.sub_dir.format(self.dump_time), self.model_base
                ),
                "model_type": self.model_type,
                "features_path": os.path.join(
                    self.sub_dir.format(self.dump_time), self.features_base
                ),
                "targets_path": os.path.join(
                    self.sub_dir.format(self.dump_time), self.targets_base
                ),
                "training_indices": i_train.tolist(),
                "test_indices": i_test.tolist(),
                "fit_kwargs": fit_kwargs,
                "predict_kwargs": predict_kwargs,
            }

            # Create sub-sub-directory using split ID.
            _dirname = self.subsub_dir.format(self.dump_time, split_id)
            os.mkdir(_dirname)

            # Save run instructions.
            with open(os.path.join(_dirname, self.in_base), "w") as f:
                json.dump(instructions, f)
            self.split_ids.append(split_id)

            # Save class in json format.
            with open(
                os.path.join(self.sub_dir.format(self.dump_time), self.validator_base),
                "w",
            ) as f:
                json.dump(self, f, cls=JobsCrossValidatorEncoder)

    def _run(self, num_subruns):
        """
        Acquire the specified compute resources and execute the
        cross_validation task.
        
        Parameters
        ---------- 
        
        num_subruns: Integer
            Number of subruns within the cross-validation job. It corresponds
            to the number of servers used.

        """

        # Assign multiple train-test splits to each subrun.
        subrun_chunks = get_chunks(self.split_ids, num_subruns)
        subrun_args = []
        for subrun_chunk in subrun_chunks:
            in_paths = []
            for split_id in subrun_chunk:
                in_paths.append(
                    os.path.join(
                        self.subsub_dir.format(self.dump_time, split_id), self.in_base
                    )
                )
            subrun_args.append({"in_paths": ":".join(in_paths)})

        # Launch parallel cross-validation.
        job_client = client("job")
        job_client.create_run(self.project_id, self.job_id, subrun_args)
        LOGGER.info(
            "Launching cross validation. "
            + "Refer to the Jobs tab for progress information"
        )

    def run(
        self,
        model,
        features,
        targets,
        split_generator,
        num_subruns,
        fit_kwargs=None,
        predict_kwargs=None,
        dump_time_format="datetime",
    ):
        """
        Write required information (model, features, targets, split indices,
        and any kwargs) to disk, and then create the job to run the
        cross-validation.
        
        Parameters
        ----------
        
        model: sklearn/keras Model
            Model to be cross-validated.
            
        features: np.array/pd.DataFrame 
            Feature matrix for training/testing.
            
        targets: np.array/pd.DataFrame 
            Target vector/matrix for training/testing.
            
        split_generator: generator 
            Generator that yields the required train/test indices upon
            iteration. It must comply with Scikit-Learn's standard for cross
            validation iterators such as `sklearn.model_selection.ShuffleSplit`.
        
        num_subruns: Integer
            Number of subruns within the cross-validation job. It corresponds
            to the number of servers used.
        
        fit_kwargs: dict, optional, default = None 
            Dictionary of any additional kwargs to be used by the model during 
            fitting.

        predict_kwargs: dict, optional, default = None
            Dictionary of any additional kwargs to be used by the model during 
            prediction.
        
        dump_time_format: String, optional, default = "datetime"
            String that controls what format to use when naming subdirectories
            which store the data written to disk. Valid options are "datetime"
            or "timestamp".
        
        """

        self._dump(
            model,
            features,
            targets,
            split_generator,
            fit_kwargs=fit_kwargs,
            predict_kwargs=predict_kwargs,
            dump_time_format=dump_time_format,
        )
        self._run(num_subruns)

    def gather(self):
        """
        Gather the results of cross validation. To enable the comparison with
        actual examples from the test dataset, this function returns not only
        the model predictions but also the `split_ids` and the `test_indices`.

        Parameters
        ----------
        
        None

        Returns
        -------
        
        split_ids: np.array
            Identifiers of train-test splits.

        test_indices: np.array
            Indices defining the test dataset.

        predictions: np.array
            Predictions of the model on the test dataset.

        """
        test_indices = []
        predictions = []
        for split_id in self.split_ids:
            subsub_dir = self.subsub_dir.format(self.dump_time, split_id)

            in_path = os.path.join(subsub_dir, self.in_base)
            with open(in_path) as f:
                data = json.load(f)
            test_indices += [data["test_indices"]]

            out_path = os.path.join(subsub_dir, self.out_base)
            if not os.path.exists(out_path):
                raise FileNotFoundError("Split not found. Job may not be finished")
            with open(out_path) as f:
                data = json.load(f)
            predictions += [data[str(split_id)]]

        return self.split_ids, np.array(test_indices), np.array(predictions)


def jobs_cross_validator_from_json(json_path):
    """
    Recreates a JobsCrossValidator object from a JSON file specifying
    its attributes.
    
    Parameters
    ----------
    
    json_path: String 
        Path to file containing JSON dictionary of required attributes.
    
    Returns
    ------- 
    
    jobs_cv: JobsCrossValidator 
        JobsCrossValidator object with the parameters specified by the 
        supplied JSON.
        
    """
    with open(json_path, "r") as f:
        raw = json.load(f)
    jobs_cv = JobsCrossValidator("", "")
    for key in jobs_cv.__dict__.keys():
        setattr(jobs_cv, key, raw[key])
    return jobs_cv
