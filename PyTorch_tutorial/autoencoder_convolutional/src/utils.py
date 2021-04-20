import os 
import numpy as np 
import imageio 
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 

import torch 
from torchvision.utils import save_image

from tqdm import tqdm 

to_pil_image = transforms.ToPILImage() 

# To create gifs from provided list of images 
def create_gifs(images_list):
    imgs = [ np.array(to_pil_image(imgs)) for imgs in images_list ]
    imageio.mimsave('../output/generated_images.gif', imgs)

# To save a reconstructed image insert the image from model to save 
def save_reconstructed_image(reconstructed_image, epoch):
    save_image(reconstructed_image.cpu(), "../output/output_{}.jpg".format(epoch))

# Plotting loss curve for the model 
def plot_loss(train_loss, validation_loss):
    fig, ax = plt.subplots(1,1, figsize=(10,8))
    ax.plot(train_loss, color='red', label='Training loss')
    ax.plot(validation_loss, color='blue', label='Validation loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    plt.legend()
    plt.show()

# LOSS FUNCTION 
def final_loss(bce_loss, mu, logvar):
    """
    This function will add the reconstruction loss (BCELoss) and the 
    KL-Divergence.
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    :param bce_loss: recontruction loss
    :param mu: the mean from the latent vector
    :param logvar: log variance from the latent vector
    """
    BCE = bce_loss 
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train(model, _dataloader, _dataset, device, _optimizer, _criterion):
    model.train()
    running_loss = 0.0
    counter = 0
    for i, data in tqdm(enumerate(_dataloader), total=int(len(_dataset)/_dataloader.batch_size)):
        counter += 1
        data = data[0]
        data = data.to(device)
        _optimizer.zero_grad()
        
        reconstruction, mu, logvar = model(data)
        
        bce_loss = _criterion(reconstruction, data)

        loss = final_loss(bce_loss, mu, logvar)
        loss.backward()
        running_loss += loss.item()
        _optimizer.step()

    train_loss = running_loss / counter 
    return train_loss

def validate(model, _dataloader, _dataset, device, _criterion):
    model.eval()
    running_loss = 0.0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(_dataloader), total=int(len(_dataset)/_dataloader.batch_size)):
            counter += 1
            data= data[0]
            data = data.to(device)
            reconstruction, mu, logvar = model(data)
            bce_loss = _criterion(reconstruction, data)
            loss = final_loss(bce_loss, mu, logvar)
            running_loss += loss.item()
        
            # save the last batch input and output of every epoch
            if i == int(len(_dataset)/_dataloader.batch_size) - 1:
                recon_images = reconstruction

    val_loss = running_loss / counter
    return val_loss, recon_images