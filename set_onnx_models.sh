#!/bin/bash

#Download and setup the enviroment to run
#onnx models

#Install git-lfs to download all onnx models from repo
pip3 install git-lfs

#Cofigure git-lfs with your local .gitconfig
git lfs install --skip-repo

#Clone the onnx-models repo in lfs format
git lfs clone git@github.com:onnx/models.git

#Note: after cloning you need to lfs fetch to get all the files locally
git lfs pull --include="*.onnx" --exclude=""

#Export the onnx model root path to be used later
cd models && export ONNX_MODEL_ROOT=$PWD

#Add path permathly to the enviroment for later
echo "ONNX_MODEL_ROOT=$ONNX_MODEL_ROOT" >> ~/.bashrc

