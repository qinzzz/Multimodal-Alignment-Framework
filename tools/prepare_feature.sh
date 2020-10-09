#!/usr/bin/env bash

# clone faster-rcnn repo

# Structure:
#
# [dir]
# |__ Multimodal_Alignment_Framework
#      |__ data
#          |__ flickr30k
# |__ faster-rcnn.pytorch


if [[ ! -d data/flickr30k ]]; then
    mkdir -p data/flickr30k
fi

cd ../
git clone https://github.com/qinzzz/faster-rcnn.pytorch/tree/pytorch-1.0
cd faster-rcnn.pytorch


# download VG labels
wget https://github.com/jwyang/faster-rcnn.pytorch/files/2355431/objects_vocab.txt


if [[ ! -d models/res101/vg ]]; then
    mkdir -p models/res101/vg
fi

# download pre-trained model on vg
# https://drive.google.com/file/d/1CS5ipXkxvjq5benNOLanQKNmVEaEtO6-/view?usp=sharing
wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1CS5ipXkxvjq5benNOLanQKNmVEaEtO6-' -O tmp.html
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(cat tmp.html | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1CS5ipXkxvjq5benNOLanQKNmVEaEtO6-" -O models/res101/vg/faster_rcnn_1_20_16193.pth
rm -rf /tmp/cookies.txt tmp.html


# run pre-trained detection model
python demo.py --net res101 --checksession 1 --checkepoch 20 --checkpoint 16193 --cuda --dataset vg


# saved files:
# obj_detection_0.1.json
# flickr30k_features.hdf5
# maf_imgid2idx.pkl

cd ../Multimodal_Alignment_Framework/data
ln -s ../../faster-rcnn.pytorch/obj_detection_0.1.json obj_detection_0.1.json

cd flickr30k
ln -s ../../../faster-rcnn.pytorch/flickr30k_features.hdf5 flickr30k_features.hdf5
ln -s ../../../faster-rcnn.pytorch/maf_imgid2idx.pkl maf_imgid2idx.pkl
wget

# generate id2idx for train, test and validation
cd ../../
python split_pkl_idx.py

echo "processing done!"