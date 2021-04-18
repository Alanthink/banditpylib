.PHONY = help docs preview reset push push_main

SPHINXBUILD   = sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

.DEFAULT: help
help:
	@echo "make docs"
	@echo "       generate new Sphinx documentation"
	@echo "make preview"
	@echo "       preview the new Sphinx documentation"
	@echo "make reset"
	@echo "       reset the documentation"
	@echo "make push"
	@echo "       push the documentation changes"
	@echo "make push_main"
	@echo "       make a new commit to the main repository"

docs:
	rm -rf build/*
	# Generate Sphinx source code
	sphinx-apidoc -M -e -o ./source ../banditpylib/
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)"
	@echo "\033[0;32mDeploying updated documentation to GitHub...\033[0m\n"
	@rm -rf site/* && cp -r build/html/* site/
	cd site && git add . && msg="rebuilding site $(date)" && git commit -m msg

preview:
	open build/html/index.html

reset:
	cd site && git reset --hard origin/master

push:
	cd site && git push

push_main:
	git add . && msg="update docs $(date)" && git commit -m msg
