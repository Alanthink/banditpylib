.PHONY = help install clean test lint fix

PYTHON = python3

.DEFAULT: help
help:
	@echo "make install "
	@echo "       install the library"
	@echo "make test"
	@echo "       run tests"
	@echo "make lint"
	@echo "       run pylint and mypy"
	@echo "make clean"
	@echo "       clean cache files"
	@echo "make fix"
	@echo "       run yapf to format all .py files"

build:
	protoc -I=banditpylib --python_out=banditpylib banditpylib/data.proto --mypy_out=banditpylib

install_requirements:
	pip install -r requirements.txt

install: install_requirements
	pip install -e .

test:
	${PYTHON} -m pytest banditpylib

lint:
	${PYTHON} -m pylint banditpylib
	${PYTHON} -m mypy banditpylib

clean-pyc:
	@find . -name '*.pyc' -delete
	@find . -name '*.pyo' -delete
	@find . -name '*~' -delete
	@find . -name '__pycache__' -type d | xargs rm -fr
	@find . -name '.pytest_cache' -type d | xargs rm -fr
	@rm -rf .mypy_cache

clean: clean-pyc
	@echo "Clean cache files"

fix:
	@yapf -irp --style="{indent_width: 2}" --exclude 'banditpylib/data_pb2.py' banditpylib
