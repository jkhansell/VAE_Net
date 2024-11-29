import torch
import torch.nn.functional as F

def KL_MSE_lossfn(reconstructed, target, mu, logvar, alpha=1.0, beta=1.0):
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
    # Mean Squared Error (Reconstruction Loss)
    mse_loss = F.mse_loss(reconstructed, target, reduction='mean')

    # KL Divergence Loss
    kl_div_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()))

    # Combined Loss
    total_loss = alpha * mse_loss + beta * kl_div_loss

    return total_loss

class VAE_Trainer():
    def __init__(model, optimizer, lossfns, lod, model_path, load_chkpt):
        self.model = model
        self.model_path = model_path
        self.optimizer = optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if load_chkpt:
            self.load_chkpt()
        
        self.model.to(device)
        self.lossfns = lossfns


    def load_chkpt(self):
        
        chkptPath = join(self.model_path, 'model-checkpoint.pt')
        chkpt = torch.load(chkptPath)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
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

    def train(self, dataloader, epochs, lod, output_iters):
        self.train_loss = np.zeros(epochs, len(self.lossfns), dtype=np.float32)
        for epoch in range(epochs):
            losses = np.zeros(len(self.lossfns), dtype=np.float32)
            batches = 0
            for batch in dataloader:
                y = self.model.forward(batch.to(device), lod)
                model.zero_grad() # zero out optimizer
                for i, value in enumerate(zip(y, self.lossfns, batch)):
                    out, lossfn, x = value
                    loss = lossfn(out, x)
                    loss.backward()
                    losses[i] += loss
            
            self.train_loss[epoch] = losses
                self.optimizer.step()

            if epoch % output_iters == 0:
                losses /= batches
                print("Epoch: {:.04f}, ELBO+MSE loss: {:.04f}".format(
                    losses[0]
                )) 
        return
    
    def evaluate(self, dataloader, lod):
        self.model.eval()
        self.eval_loss = np.zeros(len(dataloader), len(self.lossfns), dtype=np.float32)
        batches = 0
        for batch in dataloader:
            y = self.model.forward(batch.to(device), lod)
            for i, value in enumerate(zip(y, self.lossfns, batch)):
                out, lossfn, x = value
                loss = lossfn(out, x)
                loss.backward()
                losses[i] += loss
    
    def plot_losses(self):
        plt.figure(figsize=(10, 7))

        # Loss plots.
        plt.figure(figsize=(10, 7))
        plt.plot(
            self.train_loss, color='tab:blue', linestyle='-', 
            label='train loss'
        )
        plt.plot(
            self.eval_loss, color='tab:red', linestyle='-', 
            label='validataion loss'
        )
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join('output_loss.png'))