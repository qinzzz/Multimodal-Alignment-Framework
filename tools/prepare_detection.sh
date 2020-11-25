#!/usr/bin/env bash

# clone faster-rcnn repo

# Structure:
#
# [dir]
# |__ Multimodal_Alignment_Framework
#      |__ data
#          |__ flickr30k
# |__ faster-rcnn.pytorch


cd ../
git clone https://github.com/qinzzz/faster-rcnn.pytorch.git -b pytorch-1.0
cd faster-rcnn.pytorch

# build dependency
cd lib
python setup.py build develop
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'


# download VG labels
wget -P data/vg https://github.com/jwyang/faster-rcnn.pytorch/files/2355431/objects_vocab.txt


if [ ! -d models/res101/vg ]; then
    mkdir -p models/res101/vg
fi

# download pre-trained model on vg
# https://drive.google.com/file/d/1CS5ipXkxvjq5benNOLanQKNmVEaEtO6-/view?usp=sharing
wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1CS5ipXkxvjq5benNOLanQKNmVEaEtO6-' -O tmp.html
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(cat tmp.html | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1CS5ipXkxvjq5benNOLanQKNmVEaEtO6-" -O models/res101/vg/faster_rcnn_1_20_16193.pth
rm -rf /tmp/cookies.txt tmp.html

