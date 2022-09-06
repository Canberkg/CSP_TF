# Center and Scale Prediction (CSP)

This repository contains the re-implementation of Center and Scale Prediction (CSP) [1] built with Tensorflow 2+/Keras.

## Contents

* Data Preparation
* Training
* Testing
* To-do List
* Notes

## Data Preparation

CityPersons dataset is used in this repo which is a subset of Cityscapes. The following steps should be followed carefully to prepare the dataset:
1. Download the cityscapes dataset and citypersons annotations (leftImg8bit_trainvaltest.zip and gtBbox_cityPersons_trainval.zip ) from  : https://www.cityscapes-dataset.com/downloads/ 
2. Unzip these files. Set IMAGE_PATH and JSON_ANNOTATION parameters in config.py based on the path of unzipped leftImg8bit and gtBboxCityPersons respectively.
3. Set MAIN_DIR in config.py to where the project is located.
4. CityPersons dataset contains couple of subsets such as reasonable,highly_occluded etc. Set the SUBSET parameter in config.py to either reasonable, highly_occluded, small, or None. 
5. Run data_preperation.py to prepare data based on the selected subset and also it will get rid of images without annotations

## Training

Before training, the MODEL_NAME and CKPT_DIR fields in Config.py should be set.In addition, other hyperparameters such as learning rate and batch size can also be adjusted from within the same file.After that, training can be started simply by running train.py.

## Testing

For testing, the name of the model used for testing and the path to the image must be set in Config.py.. Then run test.py to get the results.

## To-Do List

- [ ] Command-Line Interface
- [ ] New Backbones (ResNet-101,etc.)


##  Notes

* Since I only have google colab pro to train and test my network, it takes a lot of time for me to train and see the results of the training. I will share the weights of the model when I succeed in obtaining a model that performs well in both training and validation data. If you have any feedback please let me know, I would love to discuss about it and improve the repository with your feedback.


## Reference
[1]LIU, Wei; HASAN, Irtiza; LIAO, Shengcai. Center and Scale Prediction: Anchor-free Approach for Pedestrian and Face Detection. arXiv preprint arXiv:1904.02948, 2019.
## Helpful Repositories 
https://github.com/WangWenhao0716/Adapted-Center-and-Scale-Prediction
