# models/discriminator.py
import torch
import torch.nn as nn

class Discriminator(nn.Module):
    """
    Adversarial discriminator model for domain adaptation in semantic segmentation.

    This is a fully convolutional network that takes a segmentation map (logits)
    as input and outputs a map of probabilities indicating whether the input
    is from the source (e.g., GTA5) or target (e.g., Cityscapes) domain.
    The architecture is based on the one proposed in "Learning to Adapt Structured
    Output Space for Semantic Segmentation" (Tsai et al., 2018).
    """
    def __init__(self, num_classes, ndf=64):
        """
        Initializes the Discriminator model.

        Args:
            num_classes (int): Number of classes in the segmentation map, which
                               corresponds to the number of input channels for
                               the discriminator. For Cityscapes/GTA5, this is 19.
            ndf (int): Number of discriminator filters in the first conv layer.
                       This controls the "width" of the discriminator network.
        """
        super(Discriminator, self).__init__()
        
        # We use a sequential model for simplicity. Each Conv2d layer downsamples
        # the input by a factor of 2.
        self.main = nn.Sequential(
            # Input shape: (batch_size, num_classes, H, W)
            nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # Shape: (batch_size, ndf, H/2, W/2)

            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # Shape: (batch_size, ndf*2, H/4, W/4)

            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # Shape: (batch_size, ndf*4, H/8, W/8)

            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # Shape: (batch_size, ndf*8, H/16, W/16)

            # Final layer to produce a single-channel prediction map.
            # No activation function here, as we will use nn.BCEWithLogitsLoss,
            # which is more numerically stable and applies the sigmoid function internally.
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1, bias=False),
            # Shape: (batch_size, 1, H_out, W_out) 
            # where H_out and W_out are slightly different from H/16 due to padding/stride
        )
        
        self.init_weights()

    def init_weights(self):
        """Initializes weights of the discriminator."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        """
        Forward pass for the discriminator.
        
        Args:
            x (torch.Tensor): The input segmentation map (logits) from the generator (BiSeNet).
                              Shape: (batch_size, num_classes, H, W).
                              
        Returns:
            torch.Tensor: A map of logits indicating the domain probability.
                          Shape: (batch_size, 1, H_out, W_out).
        """
        return self.main(x)
