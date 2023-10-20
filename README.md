# SRGAN-PyTorch
Paper Link : https://arxiv.org/pdf/1609.04802.pdf

References : 
1. https://github.com/sgrvinod/Deep-Tutorials-for-PyTorch
2. https://www.kaggle.com/code/balraj98/single-image-super-resolution-gan-srgan-pytorch/notebook
3. https://www.kaggle.com/code/luizclaudioandrade/srgan-pytorch-lightning
4. https://pytorch.org/tutorials/beginner/
5. https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/GANs
6. https://www.youtube.com/playlist?list=PLhhyoLH6IjfwIp8bZnzX8QR30TRcHO8Va
7. https://paperswithcode.com/

Concepts involved :
Residual Connections, Generative Adversarial Networks(GANs), Perceptual Loss

Note : This Model was trained on a small subset of CelebA Dataset(you can get the whole dataset from here: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) due to computational constraints.

A GAN type model is trained to convert low resolution images into high resolution. The high resolution images are used as taregts to train the Generator and as inputs for the Discriminator of SRGAN Model respectively. Low resolution versions of the high resolution images are produced by bicubic downsampling, and they are input to the Generator of SRGAN Model.

I trained on only 1000 images and you can get that custom dataset here : https://tinyurl.com/2xfe8u6j

Use 'making_lr_img.py' to create your own custom dataset of Low Resolution images. I foud it easy as compared to do it in transforms.Compose() of 
torchvision, but it's upto you.

Use 'pytorch_prediction.py' to get predicted image from the trained model. Don't forget to add the weights of your trained model in torch.load()
and the weights must be of generator only (we don't require discriminator here, it's role is over as it was only required to improve the predictions of generator model so that generator can produce something desirable and not just random noise)
