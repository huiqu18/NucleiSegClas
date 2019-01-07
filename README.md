# Joint Nuclei Segmentation and Fine-grained Classification

This repository contains the code for the paper:

Joint Segmentation and Fine-grained Classification of Nuclei in Histopathology Images, ISBI2019.


## Dependecies
Ubuntu 16.04

Pytorch 0.4.1

Python 3.6.6

## Usage

Before training, prepare training data using `prepare_data.py` and `weight_map.m`.

To training a model, set related parameters in the file `params.py` and run `python train.py`. 
The default model is ResUNet34. The user can also try other models.


To evaluate the trained model on the test set, set related parameters in the file `params.py` and 
run `python test.py`. You can also evaluate images without ground-truth labels by simply setting
`self.test['label_dir']=''` in the file `params.py` and run `python test.py`.
