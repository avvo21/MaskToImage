import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_norm_layer(channels, norm_type='batch'):
    if norm_type == 'batch':
        return nn.BatchNorm2d(channels)
    elif norm_type == 'instance':
        return nn.InstanceNorm2d(channels, affine=True)
    else:
        return None


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type='batch', dropout_p=0.0):
        super().__init__()
        layers = []

        # First Block: Conv -> Norm -> ReLU
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False))
        norm1 = get_norm_layer(out_channels, norm_type)
        if norm1:
            layers.append(norm1)
        layers.append(nn.ReLU(inplace=True))

        # Second Block: Conv -> Norm -> ReLU
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False))
        norm2 = get_norm_layer(out_channels, norm_type)
        if norm2:
            layers.append(norm2)
        layers.append(nn.ReLU(inplace=True))

        if dropout_p > 0.0:
            layers.append(nn.Dropout(dropout_p))

        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type='batch', dropout_p=0.0):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels, norm_type=norm_type, dropout_p=dropout_p)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        feat = self.conv(x) # Extract convolutional features used for skip connections
        out = self.pool(feat) # Apply max pooling to downsample spatial dimensions
        return feat, out


class Upsample(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, norm_type='batch', dropout_p=0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # The next convolution must handle: (half in_channels) + (skip channels)
        self.conv = DoubleConv((in_channels // 2) + skip_channels, out_channels, norm_type=norm_type, dropout_p=dropout_p)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            diffY = skip.size(2) - x.size(2)
            diffX = skip.size(3) - x.size(3)
            x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([skip, x], dim=1) # Concatenation with skip connection
        return self.conv(x)


class UNetGenerator(nn.Module):
    def __init__(self, args: argparse.Namespace, n_classes=19, out_channels=3):
        super(UNetGenerator, self).__init__()
        self.n_classes = n_classes

        # Retrieving parameters from args
        base_feat = args.feat
        norm_type = args.norm_type
        dropout_val = args.dropout
        mode = getattr(args, 'dropout_mode', 'full')
        n_layers = args.n_layer_g

        features = []
        for i in range(n_layers):
            channels = base_feat * (2 ** i)
            features.append(min(channels, 512)) # Block channels to max 512 (Standard Pix2Pix)

        # Layer Encoding
        self.downs = nn.ModuleList()
        in_channels = n_classes
        for feature in features:
            current_drop = dropout_val if mode == 'full' else 0.0 # In paper pix2pix is not applied to encoder
            self.downs.append(Downsample(in_channels, feature, norm_type, dropout_p=dropout_val))
            in_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2, norm_type, dropout_p=dropout_val) # Dropout in both version

        # Layer Decoding
        self.ups = nn.ModuleList()


        curr_in_channels = features[-1] * 2 # Initialize the number of channels exiting the bottleneck

        for i, feature in enumerate(reversed(features)):
            # Dropout Management
            if mode == 'full':
                current_drop = dropout_val
            elif mode == 'paper':
                is_deep_layer = i < (len(features) // 2)
                current_drop = dropout_val if is_deep_layer else 0.0
            else:
                current_drop = dropout_val

            self.ups.append(Upsample(
                in_channels=curr_in_channels,
                skip_channels=feature,
                out_channels=feature,
                norm_type=norm_type,
                dropout_p=current_drop
            ))

            curr_in_channels = feature # Update curr_in_channels for the next decoder block

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.output_activation = nn.Tanh()

    def forward(self, img):
        x = img.clone()
        x[x == 255] = 0 # Substitute temporally the pixel of the void class
        x = F.one_hot(x, num_classes=self.n_classes).permute(0, 3, 1, 2).float() # Convert in one-hot
        mask_void = (img == 255).unsqueeze(1) # Generate a binary mask
        x = x.masked_fill(mask_void, 0.0) # Set the void class in the 0 class to zero

        skip_connections = []
        for down in self.downs:
            skip, x = down(x)
            skip_connections.append(skip)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1] # Flip the skip connections list

        for idx, up in enumerate(self.ups):
            x = up(x, skip_connections[idx])

        return self.output_activation(self.final_conv(x)) # Reduce the channel to 3 and apply the activation function


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc=22, ndf=64, n_layers=3, norm_type='batch'):
        super(NLayerDiscriminator, self).__init__()
        kw = 4          # Kernel width -> dimension of the conv filter
        padw = 1        # Padding width
        sequence = []   # Empty list for storing the layers

        # First layer: No Norm to stabilize the input
        sequence += [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        nf_mult = 1 # Initializes the channel multiplier which grows dynamically
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)

            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=False)
            ]

            norm = get_norm_layer(ndf * nf_mult, norm_type)
            if norm:
                sequence.append(norm)
            sequence.append(nn.LeakyReLU(0.2, inplace=True))

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)

        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=False)
        ]

        norm_final = get_norm_layer(ndf * nf_mult, norm_type)
        if norm_final:
            sequence.append(norm_final)
        sequence.append(nn.LeakyReLU(0.2, inplace=True))

        # Single channel output for the PatchGAN
        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, inp):
        return self.model(inp)