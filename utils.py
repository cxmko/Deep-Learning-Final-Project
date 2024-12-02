import torch
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


### Function to Update M ###
def update_M(r):
    """
    Compute M = sup_x∈X p(x)/p̂(x) for a batch (max likelihood ratio).
    Args:
        D_output_fake (Tensor): Discriminator outputs for fake samples.
    Returns:
        M (float): Maximum likelihood ratio.
    """
    # Clamp discriminator outputs to avoid division by near-zero values
    M = torch.max(r).item()
    return M


### Function to Update cK ###
def update_cK(r, rejection_budget):
    """
    Compute c_K to enforce the budget constraint.
    Args:
        r (Tensor): Likelihood ratios for the generated samples.
        rejection_budget (float): Budget K for rejection sampling.
    Returns:
        c_K (float): Scaling constant for rejection sampling.
    """
    mean_likelihood_ratio = r.mean().item()
    return 1 / (mean_likelihood_ratio * rejection_budget)

def update_c_K_dichotomy(discriminator, generator, x, K, device, epsilon=1e-1):
    # Step 1: Initialize the range for c_K
    cmin = 1e-10
    cmax = 1e10

    # Step 2: Start with the midpoint of the range for c_K
    c_K = (cmax + cmin) / 2

    # Step 3: Define the loss function L(c_K)
    # a(x_fake, c_K) = min(exp(D(x_fake)) * c_K, 1) (probabilities for acceptance)
    def loss_function(c_K):
        # Generate a batch of fake samples from the generator
        z = torch.randn(x.shape[0], 100).to(device)
        generated_samples = generator(z).detach()
        generated_samples_flat = generated_samples.view(x.shape[0], -1)

        # Get the discriminator logits
        logits = discriminator(generated_samples_flat,True)

        # Calculate likelihood ratios (assumes logits are already in log space)
        likelihood_ratios = torch.exp(logits).squeeze()

        # Calculate acceptance probabilities
        acceptance_probs = torch.minimum(likelihood_ratios * c_K, torch.tensor(1.0).to(device))

        # Calculate the loss: the sum of the acceptance probabilities minus 1/K
        L_c_K = acceptance_probs.sum().item() - (x.shape[0] / K)
        return L_c_K

    # Step 4: Iteratively adjust c_K using dichotomy until the loss is within the threshold epsilon
    while True:
        L_c_K = loss_function(c_K)
        
        # Step 5: Check if the absolute value of the loss is within the tolerance
        if abs(L_c_K) < epsilon:
            break

        # Step 6 & 7: Adjust the bounds based on the sign of the loss function
        if L_c_K > 0:
            cmax = c_K
        else:
            cmin = c_K

        # Step 10: Update c_K to the midpoint of the current bounds
        c_K = (cmax + cmin) / 2

    return c_K


### Function to Compute Acceptance Probability ###
def a_O(r, cK, M):
    """
    Calculate the acceptance function a_O(x) = min(p(x)/p̂(x) * c_K / M, 1).
    Args:
        r (Tensor): Likelihood ratios.
        cK (float): Scaling constant for rejection sampling.
        M (float): Maximum likelihood ratio.
    Returns:
        Tensor: Acceptance probabilities for each sample.
    """
    return torch.minimum(r * cK / M, torch.ones_like(r))


### GAN Divergence Function ###
def f(u):
    """
    GAN divergence function: f(u) = u * log(u) - (u + 1) * log(u + 1).
    Args:
        u (Tensor): Input tensor (likelihood ratios).
    Returns:
        Tensor: Divergence values.
    """
    # Clamp u to avoid log instabilities
    clamped_u = torch.clamp(u, 1e-8, 1e6)
    return clamped_u * torch.log(clamped_u) - (clamped_u + 1) * torch.log(clamped_u + 1 + 1e-8)


### Discriminator Training ###
def D_train(x, G, D, D_optimizer, criterion, device):
    """
    Train the discriminator to distinguish real and fake samples.
    Args:
        x (Tensor): Real data samples.
        G (nn.Module): Generator model.
        D (nn.Module): Discriminator model.
        D_optimizer (Optimizer): Optimizer for the discriminator.
        criterion (Loss): Binary cross-entropy loss function.
        device (torch.device): Device for training.
    Returns:
        float: Discriminator loss.
    """
    D.zero_grad()

    # Train on real samples
    x_real, y_real = x, torch.ones(x.shape[0], 1).to(device)
    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)

    # Train on fake samples
    z = torch.randn(x.shape[0], 100).to(device)
    x_fake, y_fake = G(z), torch.zeros(x.shape[0], 1).to(device)

    D_output = D(x_fake)
    # Clamp D_output to avoid numerical instability
    D_output = torch.clamp(D_output, 1e-5, 1 - 1e-5)
    D_fake_loss = criterion(D_output, y_fake)

    # Combine losses and backpropagate
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    # Apply gradient clipping
    torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)
    D_optimizer.step()

    return D_loss.item()


### Generator Training ###
def G_train(x, G, D, G_optimizer, criterion, device, rejection_budget):
    """
    Train the generator to produce samples indistinguishable from real samples.
    Args:
        x (Tensor): Real data samples.
        G (nn.Module): Generator model.
        D (nn.Module): Discriminator model.
        G_optimizer (Optimizer): Optimizer for the generator.
        criterion (Loss): Binary cross-entropy loss function.
        device (torch.device): Device for training.
        rejection_budget (float): Budget K for rejection sampling.
    Returns:
        float: Generator loss.
    """
    G.zero_grad()

    # Generate fake samples
    z = torch.randn(x.shape[0], 100).to(device)
    G_output = G(z)
    D_output = D(G_output)

    # Calculate likelihood ratios and update M and c_K
    r = D_output / (1 - D_output + 1e-8)
    M = update_M(r)
    cK = update_cK(r, rejection_budget)
    aO = a_O(r, cK, M)  # Acceptance probabilities

    # Compute the generator loss using GAN divergence and acceptance function
    generator_loss = torch.mean(rejection_budget * aO * f(r / (rejection_budget * aO)))

    generator_loss.backward()
    # Apply gradient clipping
    torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
    G_optimizer.step()

    return generator_loss.item()


### Save Models ###
def save_models(G, D, folder):
    torch.save(G.state_dict(), os.path.join(folder, 'G.pth'))
    torch.save(D.state_dict(), os.path.join(folder, 'D.pth'))


### Load Model ###
def load_model(model, folder, model_type):
    """
    Load the state_dict from the checkpoint for the given model.
    Args:
        model (nn.Module): The model to load the weights into.
        folder (str): Directory containing the checkpoint files.
        model_type (str): Type of model ('G' for Generator, 'D' for Discriminator).
    Returns:
        nn.Module: Model with loaded weights.
    """
    if model_type == 'G':
        ckpt_path = os.path.join(folder, 'G.pth')
    elif model_type == 'D':
        ckpt_path = os.path.join(folder, 'D.pth')
    else:
        raise ValueError("Invalid model_type. Choose either 'G' for Generator or 'D' for Discriminator.")

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    return model



