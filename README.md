# ppfer

code release soon.

## Install Environment
We provide two ways to install the environment:
- Use Conda: `environments.yml`
- Use pip: `requirements.txt`

## Dataset prepare
Download two datasets for both pre-training and training.
`DFEW`: 

`CREMA-D`:

After downloading, the videos need to transform to video clips and organize as the following structure. Please note: the structure are defined in the `xxx_params.py` files. You can make changes, but ensure both same.


## Fill in all `xxx_params.py` blanks.
In each 

  
## Pre-train
There are actually has two models that need to pre-train and save the models for next step.
- Low and High frequency controllers (Two models) and Privacy leakage validator (One model) share the same pre-trained model.
- Feature compensator controller is the other model.


## Train
The training has three steps.
- Train high and low privacy enhancement networks.
- Train feature compensation network. 
- Train the video-based FER network.

