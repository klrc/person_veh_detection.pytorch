#!/bin/bash

# set paths
# sdk_path=<请填写DEngineSDK路径>
sdk_path=/nfs/opt/intellif/sdk
project_path=$(pwd)
build_time=$(date "+%Y-%m-%d-%H-%M-%S")

# docker version
version=v0.36
machine_name=tytvm
machine_id=$(docker ps -a -q -f name=${machine_name})
echo "project path: ${project_path}"

# remove existed docker container
log_highlight() {
  echo -e "\e[30;31m$1$(tput sgr0)"
}
if [[ -n ${machine_id} ]]; then
  log_highlight "Remove docker: ${machine_name}/${machine_id} ..."
  docker rm -f ${machine_id}
fi

# run docker
docker_addr=113.100.143.90:8091/dengine/tytvm:${version}
echo Run docker from ${docker_addr}
docker run -it --net=host --privileged --device /dev --name ${machine_name} \
-v $sdk_path/tytvm_v1.0.31.1:/DEngine/tytvm \
-v $sdk_path/tyassist-dev:/DEngine/tyassist \
-v $project_path:/DEngine/projects \
-w /DEngine/tytvm ${docker_addr} /bin/bash -c " \
  source env_nnp310.sh; \
  cd /DEngine/projects; \
  mkdir -p logs; \
  python3 /DEngine/tyassist/tyassist.py build --target nnp310 -c config_nnp310.yml 2>&1 | tee "logs/tyassist-build-nnp310-$build_time.log"; \
  ls\
"

