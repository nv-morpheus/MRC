#!/bin/bash

if [[ -n "${VSCODE_CONTAINER_GIT_USER}" ]] && [[ -n "${VSCODE_CONTAINER_GIT_EMAIL}" ]]
then
    echo "setting git config --global user.name ${VSCODE_CONTAINER_GIT_USER}"
    echo "setting git config --global user.email ${VSCODE_CONTAINER_GIT_EMAIL}"
    git config --global user.name ${VSCODE_CONTAINER_GIT_USER}
    git config --global user.email ${VSCODE_CONTAINER_GIT_EMAIL}
else
    echo "skipping git config setup"
    echo "set the following envs to configure git on startup: VSCODE_CONTAINER_GIT_USER, VSCODE_CONTAINER_GIT_EMAIL"
fi
