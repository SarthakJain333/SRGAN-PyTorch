# SRGAN-PyTorch
Paper Link : https://arxiv.org/pdf/1609.04802.pdf
References : 
1. https://github.com/sgrvinod/Deep-Tutorials-for-PyTorch
2. https://www.kaggle.com/code/balraj98/single-image-super-resolution-gan-srgan-pytorch/notebook
3. https://www.kaggle.com/code/luizclaudioandrade/srgan-pytorch-lightning
4. https://pytorch.org/tutorials/beginner/

Concepts involved :
Residual Connections, Generative Adversarial Networks(GANs), Perceptual Loss

Note : This Model was trained on a small subset of CelebA Dataset(you can get the whole dataset from here: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) due to computational constraints.\

A GAN type model is trained to convert low resolution images into high resolution. The high resolution images are used as taregts to train the Generator and as inputs for the Discriminator of SRGAN Model respectively. Low resolution versions of the high resolution images are produced by bicubic downsampling, and they are input to the Generator of SRGAN Model.
