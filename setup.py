from setuptools import setup

setup(
    name="faculty-xval",
    version="0.1.0",
    description=("Tools for the cross validation of machine-learning models."),
    url="https://faculty.ai",
    author="Faculty",
    author_email="info@faculty.ai",
    install_requires=[
        "click",
        "faculty",
        "keras",
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
