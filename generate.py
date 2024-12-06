import torch
import torchvision
import os
import argparse
from torchvision.utils import save_image
from model import Generator, Discriminator
from utils import load_model, update_M

def update_c_K_dichotomy(discriminator, generator, z_dim, batch_size, K, device, epsilon=1e-3):
    # Step 1: Initialize the range for c_K
    cmin = 1e-10
    cmax = 1e10

    # Step 2: Start with the midpoint of the range for c_K
    c_K = (cmax + cmin) / 2

    # Step 3: Define the loss function L(c_K)
    # a(x_fake, c_K) = min(exp(D(x_fake)) * c_K, 1) (probabilities for acceptance)
    def loss_function(c_K):
        # Generate a batch of fake samples from the generator
        z = torch.randn(batch_size, z_dim).to(device)
        generated_samples = generator(z).detach()
        generated_samples_flat = generated_samples.view(batch_size, -1)

        # Get the discriminator logits
        logits = discriminator(generated_samples_flat,True)

        # Calculate likelihood ratios (assumes logits are already in log space)
        likelihood_ratios = torch.exp(logits).squeeze()
        M=update_M(logits)

        # Calculate acceptance probabilities
        acceptance_probs = torch.minimum(likelihood_ratios * c_K/M, torch.tensor(1.0).to(device))

        # Calculate the loss: the sum of the acceptance probabilities minus 1/K
        L_c_K = acceptance_probs.sum().item() - (batch_size / K)
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



def calculate_acceptance(discriminator, samples, c_K, device):
    # Calculate acceptance probabilities using the discriminator.
    logits = discriminator(samples,True)
    likelihood_ratios = torch.exp(logits).squeeze()  # Approximate p(x) / p̂(x)

    # Optionally, scale the likelihood ratios to a reasonable range
    likelihood_ratios = torch.clamp(likelihood_ratios, min=0.01, max=10.0)  # Clamping the values
    M=update_M(likelihood_ratios)

    acceptance_probs = torch.minimum(likelihood_ratios * c_K/M, torch.tensor(1.0).to(device))
    return acceptance_probs

def obrs_sampling(generator, discriminator, z_dim, num_samples, batch_size, K, device):
    # Generate samples using OBRS.
    generator.eval()
    discriminator.eval()

    samples = []
    n_generated = 0

    # Compute c_K using dichotomy or a predefined value
    c_K = update_c_K_dichotomy(discriminator, generator, z_dim, batch_size, K, device, epsilon=1e-3) 

    while n_generated < num_samples:
        # Generate noise and samples
        z = torch.randn(batch_size, z_dim).to(device)
        generated_samples = generator(z).detach()

        # Reshape the output to (batch_size, 1, 28, 28) for MNIST images
        generated_samples = generated_samples.view(batch_size, 1, 28, 28)  # Reshape to image format

        # Normalize from [-1, 1] to [0, 1]
        generated_samples = (generated_samples + 1) / 2  # Convert to [0, 1] range

        # Flatten the images for the discriminator (convert from [batch_size, 1, 28, 28] to [batch_size, 784])
        generated_samples_flat = generated_samples.view(batch_size, -1)  # Flatten to [batch_size, 784]

        # Calculate acceptance probabilities
        acceptance_probs = calculate_acceptance(discriminator, generated_samples_flat, c_K, device)
        
        # Accept or reject samples
        accept_mask = torch.bernoulli(acceptance_probs).bool()
        accepted_samples = generated_samples[accept_mask]
        #print(len(accepted_samples)/len(generated_samples))

        # Append accepted samples
        samples.append(accepted_samples)
        n_generated += accepted_samples.size(0)

    # Concatenate all samples and limit to desired number
    samples = torch.cat(samples)[:num_samples]
    return samples



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Samples using OBRS with GAN.')
    parser.add_argument("--batch_size", type=int, default=128, help="The batch size to use for generation.")
    parser.add_argument("--num_samples", type=int, default=10000, help="The number of samples to generate.")
    parser.add_argument("--K", type=int, default=5, help="The sampling budget.")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print('Model Loading...')
    # Initialize and load the generator and discriminator models
    mnist_dim = 784  # Flattened MNIST dimensions
    z_dim = 100  # Latent dimension

    generator = Generator(g_output_dim=mnist_dim).to(device)
    discriminator = Discriminator(d_input_dim=mnist_dim).to(device)

    # Load pre-trained models
    generator = load_model(generator, 'checkpoints', "G")
    discriminator = load_model(discriminator, 'checkpoints', "D")

    generator = torch.nn.DataParallel(generator).to(device)
    discriminator = torch.nn.DataParallel(discriminator).to(device)

    print('Model loaded.')

    print('Start Generating Samples with OBRS...')
    os.makedirs('samples', exist_ok=True)

    # Generate samples using OBRS
    samples = obrs_sampling(generator, discriminator, z_dim, args.num_samples, args.batch_size, args.K, device)

    # Save the generated samples as images
    n_samples = 0
    with torch.no_grad():
        while n_samples < args.num_samples:
            x = samples[n_samples:n_samples + args.batch_size]
            for k in range(x.shape[0]):
                if n_samples < args.num_samples:
                    torchvision.utils.save_image(x[k:k+1], os.path.join('samples', f'{n_samples}.png'))
                    n_samples += 1

    print(f"Generated {n_samples} samples and saved to 'samples' directory.")

