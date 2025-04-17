import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.lbae import ImprovedLBAE, save_model_config
from train import train_model
from create_latent_ds import create_latent_mnist_dataset

def main():
    # Parameters
    batch_size = 128
    latent_dim = 20
    num_epochs = 20
    lr = 3e-4
    
    # Check for device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Get input dimensions from the dataset
    sample = next(iter(train_loader))[0]
    input_channels = sample.shape[1]
    img_size = sample.shape[2]
    
    # Initialize model
    model = ImprovedLBAE(input_channels, img_size, latent_dim).to(device)
    
    # First do a single forward pass to initialize the model
    with torch.no_grad():
        sample_batch = sample[:2].to(device)  # Just use 2 samples to initialize
        model(sample_batch)
    
    print(f"Model initialized with latent dimension: {latent_dim}")
    print(f"Total model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    model, train_losses, test_losses = train_model(
        model, train_loader, test_loader, device, num_epochs, lr)
    
    # Save model configuration
    save_model_config(model)
    
    # Create latent MNIST dataset
    print("\n====== Creating Latent MNIST Dataset ======\n")
    create_latent_mnist_dataset(model, device)
    
    print("Process complete!")

if __name__ == '__main__':
    main()
