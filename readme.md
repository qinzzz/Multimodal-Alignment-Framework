# Multimodal Alignment Framework

Implementation of MAF: Multimodal Alignment Framework for Weakly-Supervised Phrase Grounding.

Some of our code is based on [ban-vqa](https://github.com/jnhwkim/ban-vqa). Thanks!


## Prerequisites
- python 3.7
- pytorch 1.4.0 


## Data

### Flickr30k Entities
We use flickr30k dataset to train and validate our model.

Download the raw dataset [Flickr30k Entites Annotations](https://github.com/BryanPlummer/flickr30k_entities/blob/master/annotations.zip)
to `data/flickr30k/annotations.zip` manually.

Then run
`
 sh tools/prepare_data.sh
`
to downloaded and process flickr30k images and Glove word embeddings.


### Object proposals

We use an off-the-shelf [faster-rcnn](https://github.com/jwyang/faster-rcnn.pytorch) pretrained on Visual Genome 
to generate objects proposals, labels and features.

run 
`
sh tools/prepare_feature.sh
`
to perform faster-rcnn on Flickr30k Entities and generate image features. 

*NOTE: you may have to customize your environment in order to run faster-rcnn successfully. 
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
