'''
The setup.py file is used for packaging Python projects.
 It contains metadata about the project and instructions on how to install it.
'''


#  this will go through the project directory and find all packages to include __init__.py
#  and will consider them as packages to be included in the distribution
from setuptools import setup, find_packages

# typing module provides support for type hints.
from typing import List

def get_requirements()->List[str]:
    '''
    this function will return list of requirements
    '''
    requirement_list:List[str]=[]
    try:
        with open('requirements.txt', 'r') as file:
            # read lines from requirements.txt
            lines = file.readlines()
            # process each line
            for line in lines:
                requirement= line.strip()
                #  ignore empty lines and -e.
                if requirement and requirement!='-e.':
                    requirement_list.append(requirement)
    except FileNotFoundError:
        print("requirements.txt file not found.")

    return requirement_list

setup(
    name='NetworkSecurity',
    version='0.0.1',
    author='Nikita',
    author_email='nikitasinghak257@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements()
)