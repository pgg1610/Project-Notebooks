import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

import matplotlib.pyplot as plt 

import model
from utils import train, validate, save_reconstructed_image, create_gifs, plot_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize the model
model = model.ConvVAE().to(device)

# set the learning parameters
lr = 0.001
epochs = 100
batch_size = 64
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCELoss(reduction='sum')

# a list to save all the reconstructed images in PyTorch grid format
grid_images = []

#Convert original images (28 x 28) to (32 x 32)
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

# training set and train data loader
trainset = torchvision.datasets.MNIST(
    root='input', train=True, download=True, transform=transform
)

trainloader = DataLoader(
    trainset, batch_size=batch_size, shuffle=True
)

# validation set and validation data loader
valset = torchvision.datasets.MNIST(
    root='input', train=False, download=True, transform=transform
)
valloader = DataLoader(
    valset, batch_size=batch_size, shuffle=False
)

train_loss = []
valid_loss = []
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")

    train_epoch_loss = train(
        model, trainloader, trainset, device, optimizer, criterion)

    valid_epoch_loss, recon_images = validate(
        model, valloader, valset, device, criterion)

    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    
    # save the reconstructed images from the validation loop
    save_reconstructed_images(recon_images, epoch+1)

    # convert the reconstructed images to PyTorch image grid format
    image_grid = make_grid(recon_images.detach().cpu())
    grid_images.append(image_grid)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f"Val Loss: {valid_epoch_loss:.4f}")