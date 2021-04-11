# Introduction

This directory i.e., `docs` is intended for source code of `sphinx` documentation. All of the subsequent commands if not specified are assumed to be executed in this directory.

# Generate the documentation

After all the changes, run the following command to regenerate the new documentation.

```bash
./generate_doc.sh
```

This command will make a new commit to submodule `site`. You can use 

```bash
./preview.sh
```

to preview the changes before pushing the changes. If you are satisfied with the changes, you have to do `git push` under `site` directory to push the changes to remote repository.

## Undesired changes

If there are some undesired changes, please reset the submodule `site` to remote master repository i.e., 

```bash
git reset --hard origin/master
```

to avoid polluting the commit history of submodule `site` and then regenerate the documentation.

# Make commit to the main repository

Finally, do not forget to make a commit to the main repository i.e., `banditpylib` for the change of documentation.
