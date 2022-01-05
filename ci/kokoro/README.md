The entrypoint build script is gcp_ubuntu/build.sh.


To manually trigger the unit test run as the CI system does:

```
export GIT_REPO_DIR=<absolute path to the git repo>
export KOKORO_ARTIFACTS_DIR="${GIT_REPO_DIR}"
ci/kokoro/gcp_ubuntu/build.sh
```

That is, if the current directory is the root of the Git repository:
```
GIT_REPO_DIR="`pwd`" KOKORO_ARTIFACTS_DIR="`pwd`" ci/kokoro/gcp_ubuntu/build.sh
```
