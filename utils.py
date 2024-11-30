import torch
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def D_train(x, G, D, D_optimizer, criterion, device):
    #=======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real
    x_real, y_real = x, torch.ones(x.shape[0], 1).to(device)
    x_real, y_real = x_real.to(device), y_real.to(device)  # Move both to the correct device

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    # train discriminator on fake
    z = torch.randn(x.shape[0], 100).to(device)  # Move z to the correct device
    x_fake, y_fake = G(z), torch.zeros(x.shape[0], 1).to(device)

    D_output = D(x_fake)
    
    D_fake_loss = criterion(D_output, y_fake)
    D_fake_score = D_output

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()

    return D_loss.data.item()


def G_train(x, G, D, G_optimizer, criterion, device):
    #=======================Train the generator=======================#
    G.zero_grad()

    z = torch.randn(x.shape[0], 100).to(device) 
    y = torch.ones(x.shape[0], 1).to(device)

    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y)

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()

    return G_loss.data.item()

def E_train(E, cluster_ids, z, G, E_optimizer, device):
    """
    Train the encoder to predict the correct cluster for generated samples.
    """
    E.zero_grad()

    # Generate fake data from the generator
    fake_data = G(z).detach()

    # Predict cluster probabilities for fake data
    cluster_probs = E(fake_data)

    # Target clusters in one-hot encoded form
    target_clusters = torch.nn.functional.one_hot(cluster_ids, cluster_probs.size(1)).float().to(device)

    # Cross-entropy loss for encoder
    E_loss = torch.nn.functional.cross_entropy(cluster_probs, target_clusters)
    E_loss.backward()
    E_optimizer.step()

    return E_loss.item()

def save_models(G, D, folder):
    torch.save(G.state_dict(), os.path.join(folder, 'G.pth'))
    torch.save(D.state_dict(), os.path.join(folder, 'D.pth'))


import os
import torch

def load_model(model, folder, model_type):
    """
    Loads the state_dict from the checkpoint for the given model.
    
    Args:
    - model: The model (either Generator or Discriminator).
    - folder: Path to the folder containing the checkpoint files.
    - model_type: Type of model ('G' for Generator, 'D' for Discriminator). Defaults to 'G'.
    
    Returns:
    - model: The model with loaded weights.
    """
    # Determine the checkpoint file based on the model type
    if model_type == 'G':
        ckpt_path = os.path.join(folder, 'G.pth')  # For Generator
    elif model_type == 'D':
        ckpt_path = os.path.join(folder, 'D.pth')  # For Discriminator
    else:
        raise ValueError("Invalid model_type. Choose either 'G' for Generator or 'D' for Discriminator.")
    
    # Load the checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(ckpt_path, map_location=device)
    
    # Load the state_dict of the model (handling multi-GPU if applicable)
    model.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    
    return model
