#!/bin/sh

# If a command fails then the deploy stops
set -e

rm -rf build/*

sphinx-apidoc -M -e -f -o ./source ../banditpylib/

make html

printf "\033[0;32mDeploying updated documentation to GitHub...\033[0m\n"

rm -rf site/*

cp -r build/html/* site/

Commit () {
  # Add changes to git.
  git add .
  # Commit changes.
  msg="rebuilding site $(date)"
  if [ -n "$*" ]; then
    msg="$*"
  fi
  git commit -m "$msg" || true
}

# Go to site folder
cd site
Commit
