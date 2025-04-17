import os
import torch
import matplotlib.pyplot as plt
import numpy as np

def visualize_results(model, test_loader, device, num_samples=8):
    model.eval()
    
    # Get samples
    data_samples = []
    labels = []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            data_samples.append(data[:num_samples])
            labels.append(target[:num_samples])
            break
    
    data_sample = data_samples[0]
    labels = labels[0]
    
    # Get reconstructions
    with torch.no_grad():
        recon_batch, binary_latent, _ = model(data_sample)
    
    # Convert binary latents from {-1,1} to {0,1} for visualization
    binary_vis = (binary_latent + 1) / 2
    
    # Plot original, latent, and reconstructed images
    n = min(data_sample.shape[0], num_samples)
    
    fig, axes = plt.subplots(3, n, figsize=(2*n, 6))
    
    for i in range(n):
        # Original images
        axes[0, i].imshow(data_sample[i].cpu().squeeze(), cmap='gray')
        axes[0, i].set_title(f'Original #{labels[i].item()}')
        axes[0, i].axis('off')
        
        # Binary latent representation
        axes[1, i].imshow(binary_vis[i].cpu().reshape(1, -1), cmap='binary', aspect='auto')
        axes[1, i].set_title(f'Latent ({model.latent_dim} bits)')
        axes[1, i].axis('off')
        
        # Reconstructed images
        axes[2, i].imshow(recon_batch[i].cpu().squeeze().detach(), cmap='gray')
        axes[2, i].set_title('Reconstruction')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/lbae_results.png')
    plt.close()
    
    # Generate new samples from random binary latents
    visualize_random_samples(model, device, num_samples=16)
    
    return data_sample, binary_latent, recon_batch

def visualize_random_samples(model, device, num_samples=16):
    model.eval()
    
    # Generate random binary latents
    random_latents = torch.randint(0, 2, (num_samples, model.latent_dim), device=device).float() * 2 - 1
    
    # Generate samples
    with torch.no_grad():
        samples = model.decode(random_latents)
    
    # Plot samples
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    axes = axes.flatten()
    
    for i in range(num_samples):
        axes[i].imshow(samples[i].cpu().squeeze().detach(), cmap='gray')
        axes[i].axis('off')
    
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/lbae_random_samples.png')
    plt.close()

def plot_latent_mnist_samples(latents, labels, model=None, device=None, num_samples=10, output_path="plots/latent_mnist_samples.png"):
    """Plot latent MNIST samples and their reconstructions if model is provided"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Randomly select samples
    indices = np.random.choice(len(latents), num_samples, replace=False)
    
    # Extract samples
    sample_latents = latents[indices]
    sample_labels = labels[indices]
    
    # Convert from {0, 1} to {-1, 1} for model input
    binary_latents = torch.tensor(sample_latents * 2 - 1, dtype=torch.float32).to(device)
    
    # Set up the plot
    if model is None:
        # Just show latent representations
        fig, axes = plt.subplots(1, num_samples, figsize=(2*num_samples, 3))
        
        for i in range(num_samples):
            # Plot latent representation
            axes[i].imshow(sample_latents[i].reshape(1, -1), cmap='binary', aspect='auto')
            axes[i].set_title(f"Label: {sample_labels[i]}")
            axes[i].axis('off')
    else:
        # Show latent representations and reconstructions
        fig, axes = plt.subplots(2, num_samples, figsize=(2*num_samples, 5))
        
        # Reconstruct from latent
        with torch.no_grad():
            reconstructions = model.decode(binary_latents)
        
        for i in range(num_samples):
            # Plot latent representation
            axes[0, i].imshow(sample_latents[i].reshape(1, -1), cmap='binary', aspect='auto')
            axes[0, i].set_title(f"Label: {sample_labels[i]}")
            axes[0, i].axis('off')
            
            # Plot reconstruction
            axes[1, i].imshow(reconstructions[i].cpu().squeeze(), cmap='gray')
            axes[1, i].set_title("Reconstruction")
            axes[1, i].axis('off')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    
    print(f"Visualizations saved to {output_path}")

def interpolate_in_latent_space(model, img1, img2, steps=10, device='cuda'):
    """Interpolate between two images in the binary latent space"""
    model.eval()
    
    # Get latent codes for the two images
    with torch.no_grad():
        b1, _ = model.encode(img1.unsqueeze(0).to(device))
        b2, _ = model.encode(img2.unsqueeze(0).to(device))
    
    # Interpolated latents
    interpolated_images = []
    
    # Loop through steps and perform bit-wise interpolation
    for step in range(steps + 1):
        # Calculate how many bits should be flipped from b1 to b2
        num_bits_to_flip = int((step / steps) * model.latent_dim)
        
        # Create a mask of which bits to flip
        diff = (b1 != b2).float()
        idx = diff.squeeze(0).nonzero().flatten()
        
        if len(idx) > 0:
            # If we have more bits to flip than differing bits, flip all differing bits
            num_bits_to_flip = min(num_bits_to_flip, len(idx))
            
            # Create a copy of b1
            b_interp = b1.clone()
            
            # Flip the required number of bits
            if num_bits_to_flip > 0:
                idx_to_flip = idx[:num_bits_to_flip]
                b_interp[0, idx_to_flip] = b2[0, idx_to_flip]
        else:
            # No differing bits, just use b1
            b_interp = b1.clone()
        
        # Decode
        with torch.no_grad():
            recon = model.decode(b_interp)
            interpolated_images.append(recon.cpu().squeeze().numpy())
    
    return interpolated_images
