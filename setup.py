from setuptools import setup, find_packages

setup(
    name="balance_predictions",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "mlflow>=2.10.0",
        "boto3>=1.28.0",
        "pandas",
        "numpy",
        "scikit-learn"
    ]
) 