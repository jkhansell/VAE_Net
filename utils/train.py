import torch
import torch.nn.functional as F
from os.path import join
import numpy as np
import matplotlib.pyplot as plt

def logpx_z_gaussian(x, x_recon, sigma2=0.1):
    """
    Computes log p(x|z) for continuous data modeled as a Gaussian.

    Parameters:
    - x: Tensor, the original images.
    - x_recon: Tensor, the reconstructed images.
    - sigma2: float, the variance of the Gaussian distribution.

    Returns:
    - Tensor, the log-likelihood of the data given z.
    """
    var = sigma2.unsqueeze(1)
    var = var.unsqueeze(3)

    log2pi = torch.log(torch.tensor(2.0 * torch.pi))
    reconstruction_error = (x - x_recon) ** 2
    log_likelihood = -0.5 * (log2pi + torch.log(var) + reconstruction_error / var)

    return torch.sum(log_likelihood,dim=[1, 2, 3])  # Sum over image dimensions

def log_normal_pdf(sample, mean, logvar, dim=1):
    log2pi = torch.log(torch.tensor(2.0 * torch.pi))
    torch.log(torch.tensor(2. * torch.pi))
    return torch.sum(
        -0.5 * ((sample - mean)**2 * torch.exp(-logvar) + logvar + log2pi), dim=dim
        )

def KL_MSE_lossfn(output, x):
    """
    Computes the combined loss with KL divergence and Mean Squared Error.

    Parameters:
    - reconstructed: Tensor, the reconstructed output from the model.
    - target: Tensor, the ground truth or input data.
    - mean: Tensor, the mean vector from the encoder.
    - logvar: Tensor, the log-variance vector from the encoder.
    - alpha: float, weight for the MSE loss.
    - beta: float, weight for the KL divergence loss.

    Returns:
    - loss: Tensor, the total combined loss.
    """
    #print("log(p(x|z)): ",logpx_z)
    #print("log(p(z)):", logpz)
    #print("log(q(z|x)):",logqz_x)

    # Mean Squared Error (Reconstruction Loss)
    #mse_loss = (x_recon-x).pow(2).sum() / batch_size 

    # Combined Loss
    #total_loss = mse_loss + kl_div_loss
    
    x_recon, mu, logVar, z = output

    # Calculating ELBO loss    
    #logpx_z = logpx_z_gaussian(x, x_recon, torch.exp(logVar))
    #logpz = log_normal_pdf(z, torch.zeros_like(mu), torch.zeros_like(logVar))
    #logqz_x = log_normal_pdf(z, mu, logVar)
    #elbo = -torch.mean(logpx_z + logpz - logqz_x)
    # Mean Squared Error (Reconstruction Loss)
    mse_loss = F.mse_loss(x_recon, x, reduction='mean')

    # KL Divergence Loss
    kl_div_loss = torch.mean(-0.5 * torch.sum(1 + logVar - mu.pow(2) - logVar.exp()))

    return mse_loss + kl_div_loss

class VAE_Trainer():
    def __init__(self, model, optimizer, scheduler, lossfns, lod, model_path, load_chkpt, device):
        self.model = model
        self.model_path = model_path
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device 

        if load_chkpt:
            self.load_chkpt()
        
        self.model.to(self.device)
        self.lossfns = lossfns


    def load_chkpt(self):
        
        chkptPath = join(self.model_path, 'model-checkpoint.pt')
        chkpt = torch.load(chkptPath)

        self.model.load_state_dict(chkpt['model_state_dict'])
        self.optimizer.load_state_dict(chkpt['optimizer_state_dict'])
        
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)
    
    def save_chkpt(self):
        chkptPath = join(self.model_path, 'model-checkpoint.pt')
        

        torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, chkptPath)

    def train(self, dataloader, lod):
        train_losses = np.zeros(len(self.lossfns), dtype=np.float32)
        batches = 0
        for batch in dataloader:
            y = self.model.forward(batch, lod)
        
            self.model.zero_grad() # zero out optimizer

            for i, value in enumerate(zip(y, self.lossfns)):
                output, lossfn = value
                loss = lossfn(output, batch)
                loss.backward()
                train_losses[i] += loss
            
            self.optimizer.step()
            batches += 1
        
        return train_losses
    
    def evaluate(self, dataloader, lod):
        self.model.eval()
        eval_losses = np.zeros(len(self.lossfns), dtype=np.float32)
        batches = 0
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                y = self.model.forward(batch.to(self.device), lod)
                for i, value in enumerate(zip(y, self.lossfns)):                
                    output, lossfn = value
                    loss = lossfn(output, batch)
                    eval_losses[i] += loss
                batches += 1

        return eval_losses        

    def train_model(self, epochs, dataloader_train, dataloader_validate, lod, outiters): 
        
        self.train_loss = np.zeros((epochs, len(self.lossfns)), dtype=np.float32)
        self.eval_loss = np.zeros((epochs, len(self.lossfns)), dtype=np.float32)

        for epoch in range(epochs):
            self.train_loss[epoch] = self.train(dataloader_train, lod)
            self.eval_loss[epoch] = self.evaluate(dataloader_validate, lod)
        

            if epoch % int(outiters/2) == 0:
                self.scheduler.step()


            if epoch % outiters == 0:
                self.save_chkpt()
                print(f"Training loss: {self.train_loss[epoch].sum()/len(dataloader_train):.3f}")
                print(f"Validation loss: {self.eval_loss[epoch].sum()/len(dataloader_validate):.3f}")
                print(f"Learning rate: {self.optimizer.param_groups[0]['lr']}")
                print('-'*50)

            self.plot_losses(len(dataloader_train), len(dataloader_validate))


    def reconstruction(self, dataloader, lod, count):

        batch = next(iter(dataloader))
        count = min(count, len(batch))
        x = batch[:count].to(self.device)
        y = self.model.forward(x, lod)
        x_reconstruction = y[0][0]
        
        return x, x_reconstruction
    
    def save_reconstruction(self, dataloader, lod, count, outdir):
        xs, x_recons = self.reconstruction(dataloader, lod, count)
        
        for i, value in enumerate(zip(xs, x_recons)):
            x, x_recon = value

            x_cpu = x.cpu().permute(1, 2, 0).detach().numpy() * 255 
            x_recon_cpu = x_recon.cpu().permute(1, 2, 0).detach().numpy() * 255 

            x_cpu = x_cpu.astype('uint8')
            x_recon_cpu = x_recon_cpu.astype('uint8')

            # Plot the image
            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(x_cpu)
            axs[0].set_axis_off() 
            axs[1].imshow(x_recon_cpu)
            axs[1].set_axis_off() 
            fig.savefig(join(outdir, "reconstruction_"+str(i)+".png"))
            plt.close()

    def plot_losses(self, lentrain, leneval):

        # Loss plots.
        plt.figure(figsize=(10, 7))
        plt.plot(
            self.train_loss.sum(axis=1)/lentrain, color='tab:blue', linestyle='-', 
            label='train loss'
        )
        plt.plot(
            self.eval_loss.sum(axis=1)/leneval, color='tab:red', linestyle='-', 
            label='validataion loss'
        )
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.yscale("log")
        plt.legend()
        plt.savefig('output_loss.png')
        plt.close()