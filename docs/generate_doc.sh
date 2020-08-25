rm -rf build

sphinx-apidoc -M -e -f -o ./source ../banditpylib/

make html
