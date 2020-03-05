from setuptools import setup, find_packages

setup(
    name="banditpylib",
    version="0.0.2",
    author="Chester Holtz, Chao Tao",
    author_email="chholtz@eng.ucsd.edu, sdutaochao@gmail.com",
    description="A lightweight python library for bandit algorithms",
    url="https://github.com/Alanthink/banditpylib",
    packages=[package for package in find_packages()
              if package.startswith('banditpylib')],
    install_requires=[
        'absl-py',
        'attrs',
        'cvxpy',
        'cycler',
        'dill',
        'ecos',
        'future',
        'importlib-metadata',
        'kiwisolver',
        'matplotlib',
        'more-itertools',
        'multiprocess',
        'numpy',
        'osqp',
        'packaging',
        'pandas',
        'pluggy',
        'py',
        'pyparsing',
        'pytest',
        'python-dateutil',
        'pytz',
        'PyYAML',
        'scipy',
        'scs',
        'seaborn',
        'six',
        'wcwidth',
        'zipp'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6'
)
