#!/bin/sh

conda_env_find(){
    conda env list | grep "${@}" >/dev/null 2>/dev/null
}

ENV_NAME=mrc

if conda_env_find "${ENV_NAME}" ; \
then mamba env update --name ${ENV_NAME} -f ${MRC_ROOT}/ci/conda/environments/dev_env.yml; \
else mamba env create --name ${ENV_NAME} -f ${MRC_ROOT}/ci/conda/environments/dev_env.yml; \
fi

sed -ri "s/conda activate base/conda activate $ENV_NAME/g" ~/.bashrc;
