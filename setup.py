from setuptools import setup, find_packages

setup(
    name="hetpr",
    version="0.1.0",
    author="Jiahui Zhang",
    author_email="zjh68688@gmail.com",
    url="https://github.com/zjh688/Het-PR",
    description="Heteroscedastic Personalized Regression (Het-PR) runner and utilities",
    packages=find_packages(include=["model", "model.*"]), 
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn",
        "tqdm",
        "pandas",
        "pandas-plink",
        "pyranges",
        "matplotlib",
    ],
    scripts=["runHetPr.py"],
    python_requires=">=3.9",
)
