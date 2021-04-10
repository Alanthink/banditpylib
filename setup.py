import os

from setuptools import setup, find_packages

setup(name="banditpylib",
      version="1.0.0",
      author="Chester Holtz, Chao Tao, Guangyu Xi",
      author_email="chholtz@eng.ucsd.edu, sdutaochao@gmail.com, gxi@umd.edu",
      description="A lightweight python library for bandit algorithms",
      url="https://github.com/Alanthink/banditpylib/tree/dev/",
      packages=[
          package for package in find_packages()
          if package.startswith('banditpylib')
      ],
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
      ],
      python_requires='>=3.7')
