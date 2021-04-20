import torch 
import torch.nn as nn 
import torch.nn.functional as F

image_channels = 1 # MNIST dataset has grayscale digit image 
init_channels = 8 # Initial number of filters used by the ConvNet
kernel_size = 4 # Kernel size for the Conv Filter 
latent_dimensions = 16 # Dimensionality of the bottleneck layer 

# ConvVAE model 
class ConvVAE(nn.Module):
    def __init__(self):
        super(ConvVAE, self).__init__()
 
        # encoder details:
        self.enc1 = nn.Conv2d(
            in_channels=image_channels, out_channels=init_channels, kernel_size=kernel_size, 
            stride=1, padding=1)

        self.enc2 = nn.Conv2d(
            in_channels=init_channels, out_channels=init_channels*2, kernel_size=kernel_size, 
            stride=1, padding=1)

        self.enc3 = nn.Conv2d(
            in_channels=init_channels*2, out_channels=init_channels*4, kernel_size=kernel_size, 
            stride=2, padding=1)
        
        self.enc4 = nn.Conv2d(
            in_channels=init_channels*4, out_channels=64, kernel_size=kernel_size, 
            stride=2, padding=0)
        
        # fully connected layers for learning representations
        self.fc1 = nn.Linear(64, 128)
        
        self.fc_mu = nn.Linear(128, latent_dimensions)
        self.fc_log_var = nn.Linear(128, latent_dimensions)

        self.fc2 = nn.Linear(latent_dimensions, 64)

        # decoder details:
        self.dec1 = nn.ConvTranspose2d(
            in_channels=64, out_channels=init_channels*8, kernel_size=kernel_size, 
            stride=1, padding=0)

        self.dec2 = nn.ConvTranspose2d(
            in_channels=init_channels*8, out_channels=init_channels*4, kernel_size=kernel_size, 
            stride=2, padding=0)

        self.dec3 = nn.ConvTranspose2d(
            in_channels=init_channels*4, out_channels=init_channels*2, kernel_size=kernel_size, 
            stride=2, padding=0)

        self.dec4 = nn.ConvTranspose2d(
            in_channels=init_channels*2, out_channels=image_channels, kernel_size=kernel_size, 
            stride=2, padding=0)

    def reparameterize(self, mu, log_var):
        """
        INPUT:
        param mu: mean from the encoder's latent space
        param log_var: log variance from the encoder's latent space
        
        OUTPUT: 
        sample: sampled latent d
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling
        return sample
 
    def forward(self, x):
        """
        INPUT:
        x: image batch (batch, dimensions)

        OUTPUT: 
        reconstruction: reconstructed image (batch, dimensions)
        mu, log_var: sampling vectors from the bottleneck layer 
        """
        # encoding
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        batch, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        hidden = self.fc1(x)

        # get `mu` and `log_var`
        mu = self.fc_mu(hidden)
        log_var = self.fc_log_var(hidden)
        
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        z = self.fc2(z)
        print(z.shape)
        z = z.view(-1, 64, 1, 1) #Reshape the tensor 
        print(z.shape)
        # decoding
        x = F.relu(self.dec1(z))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        reconstruction = torch.sigmoid(self.dec4(x))
        return reconstruction, mu, log_var