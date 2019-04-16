![img|small](img/logo.png)
# faculty-xval
Cross validation of machine-learning models on Faculty platform. At present, the package mostly offers a way to cross validate models in parallel by means of Faculty jobs. To access the functionality one makes use of the class:

```python
faculty_xval.validation.JobsCrossValidator
```
Additional information is found in the notebooks of the `examples` directory.

The package supports `keras` and `sklearn` models. Whilst one can write custom models that are compatible with `faculty-xval`, no guarantee is given that the package handles these situations correctly, in particular because of issues concerning the randomisation of weights.

Two sets of installation instructions are provided below:
* If you would like to simply use `faculty-xval`, please follow the `User installation instructions`.
* If you would like to develop `faculty-xval` further, please follow the `Developer installation instructions`.

## User installation instructions

##### Create an environment
In your project on Faculty platform, create an environment named `faculty_xval`. In the `PYTHON` section, select `Python 3` and `pip` from the dropdown menus. Then, type `faculty-xval` in the text box, and click on the `ADD` button.

The environment installs the package `faculty-xval`, and should be applied on every server that you create; this includes both 'normal' interactive servers and job servers, as explained next.

##### Create a job definition
Create a new job definition named `cross_validation`. In the `COMMAND` section, paste the following:

`faculty_xval_jobs_xval $in_paths`

Then, add a `PARAMETER` with the name `in_paths`, and ensure that the `Make field mandatory` box is checked.

Finally, under `SERVER SETTINGS`, add `faculty_xval` to the `ENVIRONMENTS` section.

For cross-validation jobs that are computationally intensive, we recommend using dedicated servers as opposed to running in the cluster. To achieve this, click on `Large and GPU servers` under `SERVER RESOURCES`, and select an appropriate server type from the dropdown menu.

Remember to click `SAVE` when you are finished.

## Developer installation instructions

##### Select a username
Before beginning the installation process, pick an appropriate username, such as `foo`. This does not necessarily need to match your Faculty platform username. In the following instructions, your selected username will be referred to as `{USER_NAME}`.

##### Clone the repository
Create the folder `/project/{USER_NAME}`. Then, run the commands:

```bash
cd /project/{USER_NAME}
git clone https://github.com/facultyai/faculty-xval.git
```

##### Create an environment
Next, create an environment in your project named `faculty_xval_{USER_NAME}`.

In this environment, under `SCRIPTS`, paste in the following code to the `BASH` section, remembering to change the `USER_NAME` definition on the second line to your selected `{USER_NAME}`:

```bash
# Remember to change username!
USER_NAME={USER_NAME}

# Install faculty-xval from local repository.
pip install /project/$USER_NAME/faculty-xval/

# Turn USER_NAME into an environment variable.
echo "export USER_NAME=$USER_NAME" > /etc/faculty_environment.d/app.sh
if [[ -d /etc/service/jupyter ]] ; then 
  sudo sv restart jupyter
fi
```

This environment should be applied on every server that you create; this includes both 'normal' interactive servers and job servers, as explained next.

##### Create a job definition
Next, create a new job definition named `cross_validation_{USER_NAME}`. In the `COMMAND` section, paste the following:

`faculty_xval_jobs_xval $in_paths`

Then, add a `PARAMETER` with the name `in_paths`, and ensure that the `Make field mandatory` box is checked.

Finally, under `SERVER SETTINGS`, add `faculty_xval_{USER_NAME}` to the `ENVIRONMENTS` section.

For cross-validation jobs that are computationally intensive, we recommend using dedicated servers as opposed to running in the cluster. To achieve this, click on `Large and GPU servers` under `SERVER RESOURCES`, and select an appropriate server type from the dropdown menu.

Remember to click `SAVE` when you are finished.

## Try out the examples
Examples of cross validation with `faculty-xval` for the different types of model are provided in the directories `examples/keras` and `examples/sklearn`. Usage instructions are then divided in two notebooks:

* `jobs_cross_validator_run.ipynb` loads the data, instantiates the model, and starts a Faculty job that carries out the cross validation.
* `jobs_cross_validator_analyse.ipynb` gathers the results from the cross validation, reloads the target data, and calculates the model accuracy over multiple train-test splits.

Note that the example notebooks must be run in the order just defined.
