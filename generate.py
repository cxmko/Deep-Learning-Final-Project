import torch
import torchvision
import os
import argparse
from torchvision.utils import save_image
from model import Generator, Discriminator
from utils import load_model

def calculate_acceptance(discriminator, samples, c_K, device):
    # Calculate acceptance probabilities using the discriminator.
    logits = discriminator(samples)
    likelihood_ratios = torch.exp(logits).squeeze()  # Approximate p(x) / p̂(x)

    # Optionally, scale the likelihood ratios to a reasonable range
    likelihood_ratios = torch.clamp(likelihood_ratios, min=0.01, max=10.0)  # Clamping the values

    acceptance_probs = torch.minimum(likelihood_ratios * c_K, torch.tensor(1.0).to(device))
    return acceptance_probs

def obrs_sampling(generator, discriminator, z_dim, num_samples, batch_size, K, device):
    # Generate samples using OBRS.
    generator.eval()
    discriminator.eval()

    samples = []
    n_generated = 0

    # Compute c_K using dichotomy or a predefined value
    c_K = 1.0 / K  # Simplified for demonstration; tune based on acceptance rate

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
    parser.add_argument("--K", type=int, default=50, help="The sampling budget.")
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

