.PHONY = help build install clean test lint freeze fix

PYTHON = python3

.DEFAULT: help
help:
	@echo "make build"
	@echo "       generate protobuf related files"
	@echo "make install "
	@echo "       install the library"
	@echo "make test"
	@echo "       run tests"
	@echo "make lint"
	@echo "       run pylint and mypy"
	@echo "make clean"
	@echo "       clean cache files"
	@echo "make freeze"
	@echo "       generate requirements"
	@echo "make fix"
	@echo "       run yapf to format all .py files"

build:
	@echo "\033[0;32mGenerate protobuf message\033[0m"
	protoc -I=banditpylib --python_out=banditpylib banditpylib/data.proto --mypy_out=banditpylib

install_requirements:
	pip install --upgrade pip
	pip install -r requirements.txt

install: install_requirements
	pip install -e .

test:
	@echo "\033[0;32mRun tests\033[0m"
	${PYTHON} -m pytest banditpylib

lint:
	@echo "\033[0;32mCheck static errors\033[0m"
	${PYTHON} -m pylint --jobs=8 banditpylib
	@echo "\033[0;32mCheck static typing errors\033[0m"
	${PYTHON} -m mypy banditpylib

clean-pyc:
	@echo "\033[0;32mClean cache files\033[0m"
	find . -name '*.pyc' -delete
	find . -name '*.pyo' -delete
	find . -name '*~' -delete
	find . -name '__pycache__' -type d | xargs rm -fr
	find . -name '.pytest_cache' -type d | xargs rm -fr
	find . -name '.ipynb_checkpoints' -type d | xargs rm -fr
	rm -rf .mypy_cache

clean: clean-pyc

freeze:
	python3 -m pip freeze > requirements.txt

fix:
	@echo "\033[0;32mFormat code\033[0m"
	yapf -irp --style="{indent_width: 2}" --exclude 'banditpylib/data_pb2.py' banditpylib
