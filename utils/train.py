import torch
from torch.utils.data import DataLoader
import model.variational_autoencoder as VAE
import dataloader.tiffloader as TiffLoad
import os

if __name__ == "__main__":

    image_shape = (3, 2**7, 2**7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tifs = [file for file in os.listdir("./images") if file.endswith(".tif")]
    segment_size = 128  # Example: 128x128 segments
    stride = 64         # Overlap between segments
    dataset = TiffLoad.TiffDataset(os.path.join("images",tifs[0]), segment_size, stride)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    model = VAE.VariationalAutoencoder(
        img_shp=image_shape, 
        first_features_count=16, 
        activation_encoder=torch.nn.functional.elu,
        activation_decoder=torch.nn.functional.elu,
        segmentation_classes=9
    ).to(device)

    
    img = torch.randn((1, 3, 2**7, 2**7)).to(device)"""
    
