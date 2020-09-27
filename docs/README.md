# Introduction

This directory is intended for source code of documentation.

# Regenerate documentation

All the following commands are assumed to be run in directory `docs`.

After all the changes, run the following command to generate the new documentation.

```bash
./generate_doc.sh
```

This command will basically make a new commit to submodule `site`. You have to do `git push` under `site` directory to push the changes to remote repository. You can use `./preview.sh` to preview the changes. 

# Undesired documentation

If there are some undesired changes, please reset the submodule `site` to remote master repository i.e., `origin/master` to avoid several commits in submodule `site` and regenerate the documentation.
