'''
Tensor board tutorial using the NN developed for MNIST dataset 
'''
import numpy as np 
import torch 
import torch.nn as nn 
import torchvision 
import torchvision.transforms as transforms 
import matplotlib.pyplot as plt 
import matplotlib as mpl
import sys

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/mnist2")

#Device setting 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Import MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='../data/',
                                           train=True,
                                           transform=torchvision.transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../data/',
                                           train=False,
                                           transform=torchvision.transforms.ToTensor(),
                                           download=True)
input_tensor, label = train_dataset[0]
print('MNIST dataset with {} train data and {} test data'.format(len(train_dataset), len(test_dataset)))
print('Type of data in dataset: {} AND {}'.format(type(input_tensor), type(label)))
print('Input tensor image dimensions: {}'.format(input_tensor.shape))

#Model hyper-parameters for the fully connected Neural network 
input_size = 784 # Image input for the digits - 28 x 28 x 1 (W-H-C) -- flattened in the end before being fed in the NN 
num_hidden_layers = 1
hidden_layer_size = 100
num_classes = 10 
num_epochs = 10
batch_size = 100 
learning_rate = 0.001

#Convert dataset to a dataloader class for ease of doing batching and SGD operations 
from torch.utils.data import Dataset, DataLoader
train_loader = DataLoader(dataset = train_dataset,
                          batch_size = batch_size,
                          shuffle=True,
                          num_workers = 2)

test_loader = DataLoader(dataset = test_dataset,
                        batch_size = batch_size, 
                        num_workers = 2)

#Take a look at one batch 
examples = iter(train_loader)
samples, labels = examples.next()
print(samples.shape, labels.shape)

#Plotting first 4 digits in the dataset: 
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(samples[i][0], cmap=mpl.cm.binary, interpolation="nearest")
    plt.axis("off");

#plt.show()
img_grid = torchvision.utils.make_grid(samples)
writer.add_image('mnist_images',img_grid)
#writer.close()
#sys.exit()


#Define a model 
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_layer_size, num_classes):
        super(NeuralNet, self).__init__()
        self.L1 = nn.Linear(in_features = input_size, out_features = hidden_layer_size)
        self.relu = nn.ReLU()
        self.L2 = nn.Linear(in_features = hidden_layer_size, out_features = num_classes)
    
    def forward(self, x):
        out = self.relu(self.L1(x))
        out = self.L2(out) #No softmax or cross-entropy activation just the output from linear transformation
        return out


model = NeuralNet(input_size=input_size, hidden_layer_size=hidden_layer_size, num_classes=num_classes)

#Loss and optimizer 
criterion = nn.CrossEntropyLoss() #This is implement softmax activation for us so it is not implemented in the model 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

writer.add_graph(model, samples.view(-1,28*28))
#writer.close()
#sys.exit()

#Training loop 
n_total_steps = len(train_loader)

#Metrics to look at in TensorBoard 
running_loss = 0.0
running_correct = 0.0 

for epoch in range(num_epochs):
    for i, (image_tensors, labels) in enumerate(train_loader):
        #image tensor = 100, 1, 28, 28 --> 100, 784 input needed 
        image_input_to_NN = image_tensors.view(-1,28*28).to(device)
        labels = labels.to(device)
        
        #Forward pass 
        outputs = model(image_input_to_NN)
        loss = criterion(outputs, labels)
        
        #Backward 
        optimizer.zero_grad() #Detach and flush the gradients 
        loss.backward() #Backward gradients evaluation 
        optimizer.step() #To update the weights/parameters in the NN 

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        running_correct += (predicted == labels).sum().item()

        if (i+1) % 100 == 0: 
            print(f'epoch {epoch+1} / {num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')
            writer.add_scalar('training loss', running_loss / 100, epoch * n_total_steps + i )
            writer.add_scalar('accuracy', running_correct / 100, epoch * n_total_steps + i )
            running_loss = 0.0 
            running_correct = 0.0 

#Test 
with torch.no_grad():
    n_correct = 0 
    n_samples = 0 
    for images, labels in test_loader:
        images = images.view(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item() #For each correction prediction we add the correct samples 
    acc = 100 * n_correct / n_samples
    print(f'Accuracy = {acc:.2f}%')            