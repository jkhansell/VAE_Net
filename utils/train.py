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
    def __init__(model, lod, lossfns, optimizer, load_chkpt):
        self.model = model
        self.optimizer = optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if load_chkpt:
            self.load_chkpt()
        
        self.model.to(device)
        self.lossfns = lossfns


    def load_chkpt(self):
        return
    
    def save_chkpt(self):
        return

    def train(self, epochs, lod, output_iters):
        for epoch in range(epochs):
            losses = np.zeros(len(self.lossfns), dtype=np.float32)
            for batch in dataloader:
                y = self.model.forward(batch.to(device), lod)
                model.zero_grad() # zero out optimizer
                for i, value in enumerate(zip(y, self.lossfns, batch)):
                    out, lossfn, x = value
                    loss = lossfn(out, x)
                    loss.backward()
                    losses[i] += loss
                
                self.optimizer.step()
            if epoch % output_iters == 0:
                print("Epoch: {}, ELBO") 
                




    return