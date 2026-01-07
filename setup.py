from setuptools import setup, find_packages

# Minimal setup.py for packaging your model + CLI script.
# Assumes your repository structure is:
#   model/
#     __init__.py
#     BaseModel.py
#     helpingMethods.py
#     LMM.py
#     personalizedModel.py
#   runHetPr.py
#   setup.py
#
# If your folder is named "models" instead of "model", rename it or edit packages below.

setup(
    name="hetpr",
    version="0.1.0",
    author="Haohan Wang",
    author_email="haohanw@cs.cmu.edu",
    url="https://github.com/<YOUR_GITHUB_ORG>/<YOUR_REPO>",
    description="Heteroscedastic Personalized Regression (Het-PR) runner and utilities",
    packages=find_packages(include=["model", "model.*"]),  # include your model package
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn",
        "tqdm",
        # optional for prepare step:
        "pandas",
        "pandas-plink",
        "pyranges",
        "matplotlib",
    ],
    scripts=["runHetPr.py"],
    python_requires=">=3.9",
)
