import os
import datetime

from faculty import client
from keras.models import clone_model as keras_clone
from sklearn.utils.estimator_checks import check_estimator


def utc_timestamp_now():
    """
    Returns a timestamp string representing the number of milliseconds
    since 01/01/1970.
    
    Parameters
    ----------
    
    None
    
    Returns
    -------
    
    timestamp_now: String 
        String representing the current timestamp.
    
    """
    now = datetime.datetime.utcnow()
    then = datetime.datetime(1970, 1, 1)
    timestamp = (now - then).total_seconds()
    return str(int(timestamp * 1000))


def utc_datetime_now():
    """
    Returns a string representing the current datetime in a human-readable
    format.
    
    Parameters
    ----------

    None
    
    Returns
    -------
    
    utc_datetime_now: String
        String representing the current datetime in a human-readable format.
    
    """
    now = datetime.datetime.utcnow()
    return now.strftime("%Y_%m_%d_%H_%M_%S_%f")


def most_recent_xval_dirs(reference_dir, startswith="xval_", endswith="", latest=1):
    """
    Return a list of subdirectories in the reference directory that correspond
    to the latest runs of the cross-validation job. The list is sorted so that
    recent entries appear first.
    
    Parameters
    ----------
    
    reference_dir: String
        The directory where xval subdirectories are created by runs of the
        cross-validation job.
        
    startswith: String, optional, default='xval'
        String specifying the naming convention for xval subdirectories.
    
    endswith: String, optional, default = ''
        String specifying the naming convention for xval subdirectories.
        
    latest: Integer, optional, default = 1
        The number of recent runs of the cross-validation job to be considered.
        Default behaviour is to return the very latest xval subdirectory.
     
    Returns
    -------
    
    xval_dirs: List of Strings
        Paths of subdirectories containing the results from recent
        cross-validation jobs.
    
    """

    xval_dirs = []
    for name in os.listdir(reference_dir):
        path = os.path.join(reference_dir, name)
        if os.path.isdir(path):
            if name.startswith(startswith) and name.endswith(endswith):
                xval_dirs.append(path)
    return sorted(xval_dirs, reverse=True)[:latest]


def job_name_to_job_id(job_name, project_id=None):
    """
    Queries faculty platform so as to convert a specified job name into its 
    corresponding job id.

    Parameters
    ----------
    
    job_name: String
        Job name to query the platform for.
        
    project_id: uuid
        Unique id of the project on the platform.
        
    Returns
    -------
    
    job_id: uuid
        Unique job id corresponding to the specified job name and project.
    
    """

    if project_id is None:
        project_id = os.environ["FACULTY_PROJECT_ID"]
    job_client = client("job")
    for job in job_client.list(project_id):
        if job.metadata.name == job_name:
            return job.id


def keras_clone_and_compile(model):
    """
    Rebuild and recompile Keras models.

    Note
    ----
    
    The function resets the weights and the optimizer.

    Parameters
    ----------
    
    model: keras.models.Model
        A compiled Keras model.

    Returns
    -------
    
    cloned: keras.models.Model
        A copy of the input model with the weights and the optimizer being
        reset.
    
    """
    if model.optimizer is None:
        raise ValueError("Input model must be compiled.")

    # Clone model (reset weights).
    cloned = keras_clone(model)

    # Compile new model (reset optimizer).
    cloned.compile(
        optimizer=model.optimizer,
        loss=model.loss,
        metrics=model.metrics,
        loss_weights=model.loss_weights,
        sample_weight_mode=model.sample_weight_mode,
    )
    return cloned
