from model.VAE import VariationalAutoencoder as VAE
import torch

image_shape = (3, 2**7, 2**7)
vae_model = VAE(
    img_shp=image_shape,
    first_features_count=16, 
    activation_encoder=torch.nn.functional.elu,
    activation_decoder=torch.nn.functional.elu,
)

print(vae_model)