#!/usr/bin/env bash
set -e


# switch to pytorch 1.0 and install all the requirements before running this script
cd ../faster-rcnn.pytorch

# run pre-trained detection model
python demo.py --net res101 --checksession 1 --checkepoch 20 --checkpoint 16193 --cuda --dataset vg --image_dir ../Multimodal-Alignment-Framework/data/flickr30k/flickr30k_images


# saved files:
# obj_detection_0.1.json
# flickr30k_features.hdf5
# maf_imgid2idx.pkl

cd ../Multimodal_Alignment_Framework/data
ln -s ../../faster-rcnn.pytorch/obj_detection_dict_0.1.json .

cd flickr30k
ln -s ../../../faster-rcnn.pytorch/flickr30k_features.hdf5 .
ln -s ../../../faster-rcnn.pytorch/maf_imgid2idx.pkl .

# generate id2idx for train, test and validation
cd ../../
# python split_pkl_idx.py