from os import path
from setuptools import setup

# Read the contents of the README file.
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Remove logo as not supported on PyPI.
long_description = long_description.replace("![img|small](img/logo.png)", "")

setup(
    name="faculty-xval",
    version="0.1.0rc1",
    description=("Cross validation of machine-learning models on Faculty platform."),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://faculty.ai",
    author="Faculty",
    author_email="info@faculty.ai",
    install_requires=[
        "click",
        "faculty",
        "keras>=2.2.4",
        "numpy",
        "scikit_learn",
        "tensorflow",
    ],
    packages=["faculty_xval", "faculty_xval.bin"],
    entry_points={
        "console_scripts": [
            "faculty_xval_jobs_xval"
            + " = "
            + "faculty_xval.bin.jobs_cross_validation_executor:main"
        ]
    },
    zip_safe=False,
)
