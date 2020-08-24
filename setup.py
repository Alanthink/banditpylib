import os

from setuptools import setup, find_packages

lib_path = os.path.dirname(os.path.realpath(__file__))
requirement_path = os.path.join(lib_path, 'requirements.txt')
install_requires = []
if os.path.isfile(requirement_path):
  with open(requirement_path) as f:
    install_requires = f.read().splitlines()

setup(
    name="banditpylib",
    version="1.0.0",
    author="Chester Holtz, Chao Tao",
    author_email="chholtz@eng.ucsd.edu, sdutaochao@gmail.com",
    description="A lightweight python library for bandit algorithms",
    url="https://github.com/Alanthink/banditpylib/tree/dev/",
    packages=[package for package in find_packages()
              if package.startswith('banditpylib')],
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6'
)
