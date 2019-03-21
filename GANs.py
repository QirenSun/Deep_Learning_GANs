from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import torch
# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 64

# convert data to torch.FloatTensor
transform = transforms.ToTensor()

# get the training datasets
train_data = datasets.MNIST(root='data', train=True,
                                   download=True, transform=transform)

# prepare data loader
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                           num_workers=num_workers)




import matplotlib.pyplot as plt

# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy()

# get one image from the batch
img = np.squeeze(images[0])

fig = plt.figure(figsize = (3,3)) 
ax = fig.add_subplot(111)
ax.imshow(img, cmap='gray')

import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):

    def __init__(self, input_size, hidden_dim, output_size):
        super(Discriminator, self).__init__()
        
        # define all layers
        self.fc1=nn.Linear(input_size,4*hidden_dim)
        self.fc2=nn.Linear(4*hidden_dim,2*hidden_dim)
        self.fc3=nn.Linear(2*hidden_dim,hidden_dim)
        self.fc4=nn.Linear(hidden_dim,output_size)
        self.dropout=nn.Dropout(0.3)
        
    def forward(self, x):
        # flatten image
        x=x.view(-1,28*28)
        # pass x through all layers
        x=F.leaky_relu(self.fc1(x),0.2)
        x=self.dropout(x)
        x=F.leaky_relu(self.fc2(x),0.2)
        x=self.dropout(x)
        x=F.leaky_relu(self.fc3(x),0.2)
        x=self.dropout(x)
        out=self.fc4(x)
        # apply leaky relu activation to all hidden layers

        return out


class Generator(nn.Module):

    def __init__(self, input_size, hidden_dim, output_size):
        super(Generator, self).__init__()
        
        # define all layers
        self.fc1=nn.Linear(input_size,hidden_dim)
        self.fc2=nn.Linear(hidden_dim,2*hidden_dim)
        self.fc3=nn.Linear(2*hidden_dim,4*hidden_dim)
        self.fc4=nn.Linear(4*hidden_dim,output_size)
        self.dropout=nn.Dropout(0.3)        

    def forward(self, x):
        # pass x through all layers
        x=F.leaky_relu(self.fc1(x),0.2)
        x=self.dropout(x)
        x=F.leaky_relu(self.fc2(x),0.2)
        x=self.dropout(x)
        x=F.leaky_relu(self.fc3(x),0.2)
        x=self.dropout(x)
        out=F.tanh(self.fc4(x))
        return out
    
 # Discriminator hyperparams

# Size of input image to discriminator (28*28)
input_size = 784
# Size of discriminator output (real or fake)
d_output_size = 1
# Size of *last* hidden layer in the discriminator
d_hidden_size = 32

# Generator hyperparams

# Size of latent vector to give to generator
z_size = 100
# Size of discriminator output (generated image)
g_output_size = 784
# Size of *first* hidden layer in the generator
g_hidden_size = 32


# instantiate discriminator and generator
D = Discriminator(input_size, d_hidden_size, d_output_size)
G = Generator(z_size, g_hidden_size, g_output_size)

# check that they are as you expect
print(D)
print()
print(G)

# Calculate losses
def real_loss(D_out, smooth=False):
    # compare logits to real labels
    batch_size=D_out.size(0)
    # smooth labels if smooth=True
    if smooth==True:
        labels=torch.ones(batch_size)*0.9
    else:
        labels=torch.ones(batch_size)
    criterion=nn.BCEWithLogitsLoss()
    loss=criterion(D_out.squeeze(),labels)
    return loss

def fake_loss(D_out):
    # compare logits to fake labels
    batch_size=D_out.size(0)
    labels=torch.zeros(batch_size)
    criterion=nn.BCEWithLogitsLoss()
    loss = criterion(D_out.squeeze(),labels)
    return loss

import torch.optim as optim

# learning rate for optimizers
lr = 0.002

# Create optimizers for the discriminator and generator
d_optimizer = optim.Adam(D.parameters(),lr)
g_optimizer = optim.Adam(G.parameters(),lr)

import pickle as pkl

# training hyperparams
num_epochs = 40

# keep track of loss and generated, "fake" samples
samples = []
losses = []

print_every = 400

# Get some fixed data for sampling. These are images that are held
# constant throughout training, and allow us to inspect the model's performance
sample_size=16
fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
fixed_z = torch.from_numpy(fixed_z).float()

# train the network
D.train()
G.train()
for epoch in range(num_epochs):
    
    for batch_i, (real_images, _) in enumerate(train_loader):
                
        batch_size = real_images.size(0)
        
        ## Important rescaling step ## 
        real_images = real_images*2 - 1  # rescale input images from [0,1) to [-1, 1)
        
        # ============================================
        #            TRAIN THE DISCRIMINATOR
        # ============================================
                
        # 1. Train with real images

        # Compute the discriminator losses on real images
        # use smoothed labels
        d_optimizer.zero_grad()
        D_out=D(real_images)
        d_real_loss=real_loss(D_out, smooth=True)
        
        
        # 2. Train with fake images
        
        # Generate fake images
        z = np.random.uniform(-1, 1, size=(batch_size, z_size))
        z = torch.from_numpy(z).float()
        fake_images = G(z)
        
        # Compute the discriminator losses on fake images        
        D_out=D(fake_images)
        d_fake_loss=fake_loss(D_out)
                
        # add up real and fake losses and perform backprop
        d_loss = d_real_loss+d_fake_loss
        d_loss.backward()
        d_optimizer.step()
        
        # =========================================
        #            TRAIN THE GENERATOR
        # =========================================
        
        # 1. Train with fake images and flipped labels
        g_optimizer.zero_grad()
        # Generate fake images
        z=np.random.uniform(-1,1,size=(batch_size,z_size))
        z=torch.from_numpy(z).float()
        fake_images=G(z)
        # Compute the discriminator losses on fake images 
        # using flipped labels!
        D_out=D(fake_images)
        
        
        # perform backprop
        g_loss =real_loss(D_out) 
        g_loss.backward()
        g_optimizer.step()        

        # Print some loss stats
        if batch_i % print_every == 0:
            # print discriminator and generator loss
            print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                    epoch+1, num_epochs, d_loss.item(), g_loss.item()))

    
    ## AFTER EACH EPOCH##
    # append discriminator loss and generator loss
    losses.append((d_loss.item(), g_loss.item()))
    
    # generate and save sample, fake images
    G.eval() # eval mode for generating samples
    samples_z = G(fixed_z)
    samples.append(samples_z)
    G.train() # back to train mode


# Save training generator samples
with open('train_samples.pkl', 'wb') as f:
    pkl.dump(samples, f)
    
fig, ax = plt.subplots()
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator')
plt.plot(losses.T[1], label='Generator')
plt.title("Training Losses")
plt.legend()    

# helper function for viewing a list of passed in sample images
def view_samples(epoch, samples):
    fig, axes = plt.subplots(figsize=(7,7), nrows=4, ncols=4, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples[epoch]):
        img = img.detach()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.reshape((28,28)), cmap='Greys_r')   

# Load samples from generator, taken while training
with open('train_samples.pkl', 'rb') as f:
    samples = pkl.load(f)
    
# -1 indicates final epoch's samples (the last in the list)
view_samples(-1, samples)

rows = 10 # split epochs into 10, so 100/10 = every 10 epochs
cols = 6
fig, axes = plt.subplots(figsize=(7,12), nrows=rows, ncols=cols, sharex=True, sharey=True)

for sample, ax_row in zip(samples[::int(len(samples)/rows)], axes):
    for img, ax in zip(sample[::int(len(sample)/cols)], ax_row):
        img = img.detach()
        ax.imshow(img.reshape((28,28)), cmap='Greys_r')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
    
    