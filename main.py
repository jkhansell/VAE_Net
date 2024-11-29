import torch
from data.tiffdataset import TiffDataset
from utils.train import VAE_Trainer
from utils.train import KL_MSE_lossfn
from model.VAE import VariationalAutoencoder as VAE

def execute_VAE_training():

    image_shape = (3, 2**7, 2**7)
    vae_model = VAE(
        img_shp=image_shape,
        first_features_count=16, 
        activation_encoder=torch.nn.functional.elu,
        activation_decoder=torch.nn.functional.elu,
    )

    dataset = TiffDataset(data_dir="./data/images/Images_01.tif", 2**7, 64, lod=7)
    split, samples = 0.9, len(dataset)    

    dataset_train, dataset_validate = torch.utils.data.random_split(
        dataset, [int(split * samples), samples - int(split * samples)]
    )

    batch_size = 512
    epochs = 100

    dataloader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batchSize, shuffle=True, num_workers=2)
    dataloader_validate = torch.utils.data.DataLoader(dataset=dataset_validate, batch_size=batchSize, shuffle=True, num_workers=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer = VAE_Trainer(vae_model, optimizer, [KL_MSE_lossfn], 7, load_chkpt=True) 

    trainer.train(dataloader_train, epochs, lod, 500)
    trainer.evaluate(dataloader_validate, lod)
    trainer.save_chkpt()
    
