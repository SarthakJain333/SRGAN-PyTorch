import gradio as gr 
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision.utils import save_image

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, 0.8),
            nn.PReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, 0.8),
        )

    def forward(self, x):
        return x + self.conv_block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=16):
        super(GeneratorResNet, self).__init__()

        # First layer
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=9, stride=1, padding=4), nn.PReLU())

        # Residual blocks
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(64))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64, 0.8))

        # Upsampling layers
        upsampling = []
        for out_features in range(2):
            upsampling += [
                # nn.Upsample(scale_factor=2),
                nn.Conv2d(64, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.PixelShuffle(upscale_factor=2),
                nn.PReLU(),
            ]
        self.upsampling = nn.Sequential(*upsampling)

        # Final output layer
        self.conv3 = nn.Sequential(nn.Conv2d(64, out_channels, kernel_size=9, stride=1, padding=4), nn.Tanh())

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        return out
    
generator = GeneratorResNet()

# change the map_location if you're GPU rich ;-)
generator.load_state_dict(torch.load('generatorfinal.pth', map_location=torch.device('cpu')))

# https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch
generator.eval() 

def pytorch_predict(image): 
    # change the image path 
    # image = Image.open(img_path)
    # image = image.convert('RGB')
    image = torch.from_numpy(np.array(image, dtype=np.float32))
    image = image.permute(2, 0, 1)
    image = image / 255.0
    # Pass the preprocessed image to the model.
    predictions = generator(image.unsqueeze(0))
    # print(predictions)
    # print(predictions.size())
    save_image(predictions, 'result.jpg', normalize=True)
    return 'result.jpg'

gr.Interface(
    pytorch_predict,
    inputs='image',
    outputs='image',
).launch()
