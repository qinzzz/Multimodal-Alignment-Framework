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

We use an off-the-shelf [faster-rcnn](https://github.com/jwyang/faster-rcnn.pytorch) pretrained on Visual Genome 
to generate objects proposals and labels. 
We use [Bottom-Up Attention](https://github.com/airsplay/py-bottom-up-attention) for visual features.

As [Issue#1](https://github.com/qinzzz/Multimodal-Alignment-Framework/issues/1#issue-727382153) pointed out, there is some inconsistency
between features generated using our script (faster-rcnn) and Bottom-Up Attention.
We therefore upload our generated features.

Download [test features.hdf5](https://drive.google.com/file/d/1Uwv5S8qPp0rkCtR2bD8PNiYsJ0WL-u5a/view?usp=sharing) and [train_feature.hdf5](https://drive.google.com/file/d/1zxghit_mDyIKhZRemN6EDCZ3xMR4xPu5/view?usp=sharing)(20GB) to `data/flickr30k`.

Download [test_detection_dict.json](https://drive.google.com/file/d/1-r4u45EyxY7uaIk6VxCZxCiBxaOlaTC2/view?usp=sharing) and [train_detection_dict.json](https://drive.google.com/file/d/1_S-zyKF7F8SIEht6V66Sqbsz9TBqzY-P/view?usp=sharing) to `data/`.

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

