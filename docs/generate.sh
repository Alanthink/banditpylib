# sphinx-apidoc -M -e -f -t apidoc/ -o ./source ../banditpylib/
cd docs/
make clean
make html
