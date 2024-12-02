import torch
import os
from tqdm import trange
import argparse
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from model import Generator, Discriminator
from utils import D_train, G_train, update_cK, save_models

# Function to plot and save losses
def plot_losses(G_losses, D_losses, filename='losses.png'):
    plt.figure(figsize=(10, 5))
    plt.plot(G_losses, label="Generator Loss")
    plt.plot(D_losses, label="Discriminator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("GAN Training Losses")
    plt.savefig(filename)
    plt.close()

def reset_weights(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser(description='Train GAN with OBRS.')
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs for training.")
    parser.add_argument("--lr", type=float, default=0.0002,
                        help="The learning rate to use for training.")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Size of mini-batches for SGD.")
    parser.add_argument("--rejection_budget", type=float, default=2.0,
                        help="Rejection sampling budget K (default: 2).")

    args = parser.parse_args()

    # Create directories for checkpoints and data
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    # Check for CUDA availability and set the device accordingly
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    print(f"Using device: {device}")

    # Data Pipeline
    print('Dataset loading...')
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5), std=(0.5))])

    train_dataset = datasets.MNIST(root='data/MNIST/', train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=args.batch_size, shuffle=True)
    print('Dataset Loaded.')

    # Model Loading
    print('Model Loading...')
    mnist_dim = 784
    G = Generator(g_output_dim=mnist_dim).to(device)
    reset_weights(G)
    D = Discriminator(mnist_dim).to(device)
    reset_weights(D)
    print('Model loaded.')

    # Define loss function
    criterion = nn.BCELoss()

    # Define optimizers
    G_optimizer = optim.Adam(G.parameters(), lr=args.lr,betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr=0.00002,betas=(0.5, 0.999))

    print('Start Training:')
    
    # Training loop
    G_losses = []
    D_losses = []  # Initial value for cK
    for epoch in trange(1, args.epochs + 1, leave=True, desc="Epoch Progress"):
        G_epoch_loss, D_epoch_loss = 0.0, 0.0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(-1, mnist_dim).to(device)

            # Train Discriminator
            D_loss = D_train(x, G, D, D_optimizer, criterion, device)
            D_epoch_loss += D_loss

            # Train Generator with OBRS and GAN Divergence
            G_loss = G_train(x, G, D, G_optimizer, criterion, device, args.rejection_budget)
            G_epoch_loss += G_loss

        # Average losses for this epoch
        G_losses.append(G_epoch_loss / len(train_loader))
        D_losses.append(D_epoch_loss / len(train_loader))

        print(f"Epoch {epoch}/{args.epochs}: G_loss: {G_losses[-1]}, D_loss: {D_losses[-1]}")

        # Save model checkpoints every 10 epochs
        if epoch % 10 == 0:
            save_models(G, D, 'checkpoints')

    # Plot losses after training
    plot_losses(G_losses, D_losses)
    print('Training done.')

