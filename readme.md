# Multimodal Alignment Framework

Implementation of MAF: Multimodal Alignment Framework for Weakly-Supervised Phrase Grounding.

Some of our code is based on [ban-vqa](https://github.com/jnhwkim/ban-vqa). Thanks!

**TODO**
provide Faster R-CNN feature extraction script.


## Prerequisites
- python 3.7
- pytorch 1.4.0 


## Data

### Flickr30k Entities
We use flickr30k dataset to train and validate our model.

the raw dataset can be found at [Flickr30k Entites Annotations](https://github.com/BryanPlummer/flickr30k_entities/blob/master/annotations.zip)

Run
`
 sh tools/prepare_data.sh
`
to downloaded and process Flickr30k Annotations, Images and Glove word embeddings.


### Object proposals

#### Donwload object proposals:

We use an off-the-shelf [faster-rcnn](https://github.com/jwyang/faster-rcnn.pytorch) pretrained on Visual Genome 
to generate objects proposals and labels. 
We use [Bottom-Up Attention](https://github.com/airsplay/py-bottom-up-attention) for visual features.

As [Issue#1](https://github.com/qinzzz/Multimodal-Alignment-Framework/issues/1#issue-727382153) pointed out, there is some inconsistency
between features generated using our script (faster-rcnn) and Bottom-Up Attention.
We therefore upload our generated features.

Download [train_features_compress.hdf5](https://drive.google.com/file/d/1ABnF0SZMf6pOAC89LJXbXZLMW1X86O96/view?usp=sharing)(6GB), [val features_compress.hdf5](https://drive.google.com/file/d/1iK-yz6PHwRuAciRW1vGkg9Bkj-aBE8yJ/view?usp=sharing), and [test features_compress.hdf5](https://drive.google.com/file/d/1pjntkbr20l2MiUBVQLVV6rQNWpXQymFs/view?usp=sharing) to `data/flickr30k`.

alternative link for train_feature.hdf5 (20GB, same features): [google drive](https://drive.google.com/file/d/1zxghit_mDyIKhZRemN6EDCZ3xMR4xPu5/view?usp=sharing); [baidu drive](https://pan.baidu.com/s/1cyiKNYZzpja-5brcn9QD1A), code: n1yd.

Download [train_detection_dict.json](https://drive.google.com/file/d/1_S-zyKF7F8SIEht6V66Sqbsz9TBqzY-P/view?usp=sharing), [val_detection_dict.json](https://drive.google.com/file/d/1KmyG0mghwydkb7pEwxDjItwZvNi_DRA4/view?usp=sharing), and [test_detection_dict.json](https://drive.google.com/file/d/1-r4u45EyxY7uaIk6VxCZxCiBxaOlaTC2/view?usp=sharing) and  to `data/`.

#### Generate object proposals by yourself(TODO)

~~run ` sh tools/prepare_detection.sh ` to clone faster-rcnn code and download pre-trained models.~~

~~run ` sh tools/run_faster_rcnn.sh ` to run faster-rcnn detection on flickr30k dataset and generate features.~~

*you may have to customize your environment in order to run faster-rcnn successfully. 
See [prerequisites](https://github.com/jwyang/faster-rcnn.pytorch#prerequisites)*


## Training

`
python main.py [args]
`

In our experiments, we get a ~61% accuracy using the default setting.


## Evaluating

Our trained model can be downloaded at [google drive](https://drive.google.com/file/d/1hVLDcsks2MuDJWpl2QB1H8DBCUefKCRY/view?usp=sharing).

`
python test.py --file <saved model>
`

