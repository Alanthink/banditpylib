# Introduction

This directory i.e., `docs` is intended for source code of `sphinx` documentation.

# Generate the Sphinx Documentation

After all the changes, run the following command to generate the new documentation.

```bash
make docs
```

This command will make a new commit to submodule `site`. You can use 

```bash
# open file build/html/index.html
make preview
```

to preview the changes before pushing the changes. 

If there are some undesired changes, please do 

```bash
make reset
```

to reset the submodule `site` such that the commit history of submodule `site` is not polluted.

If you are satisfied with the changes, you can do 

```bash
make push
```

to push the changes to remote repository for submodule `site`. 

Finally, do not forget to make a new commit to the main repository i.e., `banditpylib` to record this update.
