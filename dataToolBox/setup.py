from setuptools import setup, find_packages

setup(
    name="dataToolBox",
    version="1.0",
    packages=find_packages(include=["dataToolBox", "dataToolBox.*"]),
    install_requires=["sqlalchemy", "pandas", "logging"],
    description="Un module pour traiter des données en python",
    author="Quentin Samudio",
    author_email="quentin.samudio@yahoo.fr",
)
