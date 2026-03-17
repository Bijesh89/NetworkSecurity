"""
this is the setup file for the NetworkSecurity package. It defines the package metadata and dependencies.
it is used to install the package and its dependencies using pip. The setup function is called with the package name, version, description, author, and other metadata.
It also specifies the packages to include and the dependencies required for the package to work.
"""
from setuptools import setup, find_packages
from typing import List

def get_requirements() -> List[str]:
    """
    This function reads the requirements.txt file and returns a list of dependencies.
    """
    requirement_lst = []
    try:
        with open('requirements.txt', 'r') as file:
            # read lines from the file
            lines = file.readlines()
            # process each line to get the dependency
            for line in file:
                requirement = line.strip()
                if requirement and requirement[0] != '-e .': 
                     # ignore empty lines and comments
                    requirement_lst.append(requirement)
    except FileNotFoundError:
        print("requirements.txt file not found. Please make sure it exists.")

    return requirement_lst

print(get_requirements())
