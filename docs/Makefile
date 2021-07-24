.PHONY = help docs preview reset push push_main

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
	@echo "       push the documentation changes to the main repository"

docs:
	@echo "\033[0;32mDelete previous built files\033[0m"
	rm -rf build/*
	@echo "\n\033[0;32mRun sphinx\033[0m"
	sphinx-build -M html source build
	@echo "\n\033[0;32mCopy example image\033[0m"
	cp ../example.jpg build/html/
	@echo "\n\033[0;32mCopy built files to submodule\033[0m"
	rm -rf site/* && cp -r build/html/* site/
	@echo "\n\033[0;32mDeploy updated documentation to GitHub\033[0m"
	cd site && git add .
	cd site && git commit -m "rebuilding site $$(date)"

preview:
	open build/html/index.html

reset:
	cd site && git reset --hard origin/master

push:
	cd site && git push

push_main:
	git add . && git commit -m "docs update" && git push