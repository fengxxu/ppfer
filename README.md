# ppfer
code for paper "Facial Expression Recognition with Controlled Privacy Preservation and Feature Compensation"


## Install Environment
We provide two ways to install the environment:
- Use Conda: `environments.yml`
- Use pip: `requirements.txt`

## Dataset prepare
Download two datasets for both pre-training and training.

`DFEW`: 
https://dfew-dataset.github.io/

`CREMA-D`:
https://github.com/CheyneyComputerScience/CREMA-D

After downloading, the videos need to transform to video clips and organize as the required structure. 
Please note: the structure are defined in the `xxx_params.py` files.



## Fill in all `xxx_params.py` blanks.
Please follow the comments in `xxx_params.py` file to fill in blanks according your machine, file system, environment and operating system.

  
## Pre-train
There are actually has three models that need to pre-train and save the models for next step.
- Privacy Enhancer
- Low and High frequency controllers (Two models) and Privacy leakage validator (One model) share the same pre-trained model.
- Feature compensator controller is the other model.


## Train
The training has three steps.
- Train high and low privacy enhancement networks - s1.
- Train feature compensation network - s2. 
- Train the video-based FER network - s3.

