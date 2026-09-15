from setuptools import find_packages
from setuptools import setup

setup(
    name='robot_3dof',
    version='0.1.0',
    packages=find_packages(
        include=('robot_3dof', 'robot_3dof.*')),
)
