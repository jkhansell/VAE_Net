import torch
from data.tiffdataset import TiffDataset
from utils.train import VAE_Trainer
from utils.train import KL_MSE_lossfn
from model.VAE import VariationalAutoencoder as VAE

import torch.multiprocessing as mp

#mp.set_start_method('spawn', force=True)


def execute_VAE_training():

    image_shape = (3, 2**7, 2**7)
    vae_model = VAE(
        img_shp=image_shape,
        first_features_count=16, 
        activation_encoder=torch.nn.functional.elu,
        activation_decoder=torch.nn.functional.elu,
    )

    lod=7
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TiffDataset("./data/images/image_01.tif", 2**7, 64, device=device, lod=lod)
    split, samples = 0.9, len(dataset) 

    dataset_train, dataset_validate = torch.utils.data.random_split(
        dataset, [int(split * samples), samples - int(split * samples)]
    )

    batch_size = 64
    epochs = 200

    dataloader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
    dataloader_validate = torch.utils.data.DataLoader(dataset=dataset_validate, batch_size=batch_size, shuffle=True, num_workers=0)

    optimizer = torch.optim.Adam(vae_model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85)

    trainer = VAE_Trainer(vae_model, optimizer, scheduler, [KL_MSE_lossfn], 7, "./model/", load_chkpt=False, device=device) 

    trainer.train_model(epochs, dataloader_train, dataloader_validate, lod, 10)
    trainer.save_reconstruction(dataloader_train, lod, 10, "./reconstructions")
    # epochs, dataloader_train, dataload_eval, lod, outiters
    trainer.save_chkpt()
    
if __name__ == "__main__":
    execute_VAE_training()