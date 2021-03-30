# DEEP_IMPLEMENTS
This Project is where I Implement popular Deep Neural Network architecture from scratch using pytorch. 
This project is largely designed based on pytorch libraries such as learning the classification architectures which are given on torchvision library only.

## Contents
- IMAGE DATA
  - LeNet (1998)
  - AlexNet (2012)
  - ZFNet / Clarifai (2013)
  - VGGNET (2014) 
  - GoogLeNet (2014) 
    - with inception units
  - Residual Network (ResNet in 2015) 
  - Densely Connected Network (DenseNet) (2017)
  - FractalNet (2016) 
    - Alternate to ResNet
  - CapsuleNet
- Applications of CNNs
  - CNNs for solving Graph problem
  - Image processing and computer vision
  - Speech processing
  - CNN for medical imaging
- SEQUENCE DATA
  - RECURRENT NEURAL NETWORKS (RNN)
  - Long Short Term Memory (LSTM)
  - Gated Recurrent Unit (GRU)
  - Convolutional LSTM (ConvLSTM)
  - Attention based models with RNN 
  - Attention
  - Transformers
- AUTO-ENCODER (AE) AND RESTRICTED BOLTZMANN
MACHINE (RBM)
  - Review of Auto-Encoder (AE)
  - Variational auto encoders (VAEs)
  - Split-Brain Auto-encoder
- Applications of AE
  - Bio-informatics, cyber security,  
  - We can apply AE for unsupervised feature extraction and then apply Winner Take All (WTA) for clustering those samples for generating labels. 
  - AE has been used as a encoding and decoding technique with or for other deep learning approaches including CNN, DNN, RNN and RL in the last decade.
- GENERATIVE ADVERSARIAL NETWORKS (GAN)
  - Review on GAN
- Applications of GAN
  - GAN for image processing
  - GAN for speech and audio processing
  - GAN for medical information processing
  - Other applications
    - Bayesian Conditional GAN (BC-GAN)
    - Checkov GAN
    - MMD-GAN
    - MMD-GAN approach significantly outperforms Generative moment matching network (GMMN) technique which is an alternative approach for generative model
- DEEP REINFORCEMENT LEARNING (DRL)
  - Review on DRL
  - Q- Learning
- Applications of DRL
  - 

## How to use it
 There are two main folders `src` and `notebooks`. The `notebooks` has jupyter notebooks of implementations and the `src` as python files.  
 - I have implemented first on notebooks then on python files
## Image Data
Image data based networks. CNN based DNN architectures.

### LeeNet:

### Alexnet:

### VGG16:

### RESNET:

### Inception:

## Sequence Data
Sequence data types.

### RNN, LSTM, GRU n all:

### Attention:

### Transformers:


## Requirements
- conda environment create  
  `conda create -n dl python==3.8`

- Update conda env using env_dl.yml  
  `conda activate dl`  
  `conda env update -f env_dl.yml`

## Local install
- setuptools
- To build the package use  
`python setup.py sdist bdist_wheel`

- To locally install type below command from the directory where setup.py is present.  
`pip install -e .` 