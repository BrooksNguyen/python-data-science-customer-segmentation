from setuptools import setup, find_packages

setup(
    name="clustering-pipeline",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.24.0',
        'pandas>=2.0.0',
        'scikit-learn>=1.3.0',
        'matplotlib>=3.7.0',
        'seaborn>=0.12.0',
        'scipy>=1.10.0',
        'joblib>=1.2.0',
    ],
    extras_require={
        'bayesian': ['optuna>=3.0.0'],
        'full': ['optuna>=3.0.0', 'category-encoders>=2.6.0'],
    },
)
