#!/bin/bash

pip install -r requirements.txt

if [ ! -d syncnet_python ]; then
  git clone https://github.com/joonson/syncnet_python.git
  cd syncnet_python/
  bash download_model.sh
  sed -i '/PATH_WEIGHT =/s@./detectors@./syncnet_python/detectors@' detectors/s3fd/__init__.py
  cd -
fi

