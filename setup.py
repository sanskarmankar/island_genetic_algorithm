from setuptools import setup, find_packages

setup(
    name="island_ga",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas"
    ],
    python_requires=">=3.8",
    author="Sanskar Mankar",
    description="Island Genetic Algorithm for parameter optimization",
)
