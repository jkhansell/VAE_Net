import torch
import torch.nn as nn
import numpy as np

# Partially based on the autoencoder in 
# https://github.dev/jgwiese/gdemo_pancreassegmentation/blob/master/src/Autoencoder.py


class VariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder implementation with modular encoder and decoder.
    Allows flexibility in levels of detail and latent space size.
    """
    def __init__(self, img_shp, first_features_count, activation_encoder, activation_decoder):
        super(VariationalAutoencoder, self).__init__()

        self.img_shp = img_shp
        self.first_features_count = first_features_count

        self.encoder = VariationalEncoder(
            img_shp=img_shp,
            first_features_count=first_features_count,
            activation=activation_encoder
        )

        self.decoder = VariationalDecoder(
            img_shp=img_shp,
            first_features_count=first_features_count,
            activation=activation_decoder,
            segmentation=False
        )

    def forward(self, x, level_of_detail):
        """
        Forward pass for the VAE. Encodes the input, samples from the latent space,
        and decodes the result.

        Args:
            x: Input tensor.
            level_of_detail: Depth of encoding/decoding (number of layers).

        Returns:
            x_reconstructed: Reconstructed input.
            mu: Mean of latent distribution.
            logVar: Log variance of latent distribution.
        """
        level_of_detail -= 2
        z, mu, logVar, shape = self.encoder.forward(x, level_of_detail)
        x_reconstructed = self.decoder.forward(z, level_of_detail, shape)

        return ((x_reconstructed, mu, logVar, z),)

class VariationalEncoder(nn.Module):
    """
    Encoder module for the VAE, implementing the reparameterization trick.
    """
    def __init__(self, img_shp, first_features_count, activation):
        super(VariationalEncoder, self).__init__()

        self.kernel_size = 4
        self.stride = 2
        self.padding = 1

        self.img_shp = img_shp
        self.first_features_count = first_features_count

        self.activation = activation
    
        # Encoding layer lists, there are layers for the downsampling process of the image
        # and layers for latent distribution approximation of the form N(µ,σ)

        self.encoding_layers = nn.ModuleList()
        self.mu_layers = nn.ModuleList()
        self.logVar_layers = nn.ModuleList()
    
        # Get current variables to calculate latent space size
        current_channels = self.img_shp[0]
        current_height, current_width = self.img_shp[1], self.img_shp[2]

        for level in range(int(np.log2(self.img_shp[1])-1)):
            if level == 0:
                out_channels = self.first_features_count
            else:
                out_channels = self.first_features_count*2**(level)

            self.encoding_layers.append(
                nn.Conv2d(
                    # makes sure that img_shp decreases while increasing feature 
                    # extraction
                    in_channels=current_channels, 
                    out_channels=out_channels, 
                    kernel_size=self.kernel_size, 
                    stride=self.stride, 
                    padding=self.padding
                )
            )

            # Update current shape
            current_channels = out_channels
            current_height = (current_height - self.kernel_size + 2*self.padding) // self.stride + 1
            current_width = (current_width - self.kernel_size + 2*self.padding) // self.stride + 1

            in_features = current_channels*current_height*current_width
            code_length = int(2**(level+2))

            self.mu_layers.append(nn.Linear(in_features=in_features, out_features=code_length))
            self.logVar_layers.append(nn.Linear(in_features=in_features, out_features=code_length))
        
    def sample_distribution(self, mu, logVar):
        # Use the reparametrization trick for backpropagation
        std = torch.exp(0.5 * logVar)
        epsilon = torch.randn_like(std) # introduce noise (variation) 
        z = mu + std*epsilon            # calculate latent variable vector
        return z

    def encode(self, x, level_of_detail):
        # Apply encoder functionality
        
        for layer in range(level_of_detail):
            x = self.activation(self.encoding_layers[layer](x))
            
        x = self.encoding_layers[level_of_detail](x)

        # Variation Encoder
        shape = x.shape
        x = torch.flatten(x, start_dim=1)
        mu = self.mu_layers[level_of_detail](x)
        logVar = self.logVar_layers[level_of_detail](x)

        return mu, logVar, shape

    def forward(self, x, level_of_detail):
        mu, logVar, shape = self.encode(x, level_of_detail)
        z = self.sample_distribution(mu, logVar)
        return z, mu, logVar, shape

class VariationalDecoder(nn.Module):
    def __init__(self, img_shp, first_features_count, activation, segmentation=False, segmentation_classes=0):
        super(VariationalDecoder, self).__init__()
    
        self.kernel_size = 4
        self.stride = 2
        self.padding = 1

        self.segmentation = segmentation
        self.segmentation_classes = segmentation_classes

        self.img_shp = img_shp
        self.first_features_count = first_features_count

        self.activation = activation # still don't know if this is a good choice
        self.decoding_layers = nn.ModuleList()
        self.zDecodingLayers = nn.ModuleList()

        current_channels = self.img_shp[0]
        current_height, current_width = self.img_shp[1], self.img_shp[2]

        for level in range(int(np.log2(self.img_shp[1])-1)):
            if level == 0:
                out_channels = self.first_features_count
            else:
                out_channels = self.first_features_count*2**(level)

            self.decoding_layers.append(
                nn.ConvTranspose2d(
                    in_channels=out_channels,
                    out_channels=current_channels,
                    kernel_size=self.kernel_size, 
                    stride=self.stride, 
                    padding=self.padding
                )
            )

            current_channels = out_channels
            current_height = (current_height - self.kernel_size + 2*self.padding) // self.stride + 1
            current_width = (current_width - self.kernel_size + 2*self.padding) // self.stride + 1
        
            in_features = current_channels*current_height*current_width
            code_length = int(2**(level+2))

            self.zDecodingLayers.append(nn.Linear(in_features=code_length, out_features=in_features))

        if self.segmentation:
            self.segmentation_layer = nn.Conv2d(
                in_channels=current_channels, 
                out_channels=self.segmentation_classes,
                kernel_size=1
            )
        

    def decode(self, z, level_of_detail, shape):
        x = self.activation(self.zDecodingLayers[level_of_detail](z)).reshape(shape)
        
        # Transpose Convolutions
        for layer in range(level_of_detail):
            x = self.activation(self.decoding_layers[level_of_detail-layer](x))
            

        if self.segmentation:
            x = self.activation(self.segmentation_layer(self.decoding_layers[0](x)))
        else:
            x = self.decoding_layers[0](x) 
        return x

    def forward(self, z, level_of_detail, shape, detach=False):
        if detach:
            z = z.detach()
        x = self.decode(z, level_of_detail, shape)
        return x