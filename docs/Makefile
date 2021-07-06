.PHONY = help docs preview reset push

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

docs:
	# Delete previous built files
	rm -rf build/*
	@sphinx-build -M html source build
	@cp ../example.jpg build/html/
	@echo "\033[0;32mDeploying updated documentation to GitHub...\033[0m\n"
	@rm -rf site/* && cp -r build/html/* site/
	cd site && git add .
	cd site && git commit -m "rebuilding site $$(date)"

preview:
	open build/html/index.html

reset:
	cd site && git reset --hard origin/master

push:
	cd site && git push
