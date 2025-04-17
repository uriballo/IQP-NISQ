import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.lbae import load_trained_model, save_model_config
from datasets.latent_dataset import process_dataset

def create_latent_mnist_dataset(model=None, device=None, output_dir="latent_mnist_data"):
    """Create latent MNIST dataset from a pre-trained LBAE model"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Step 1: Loading pre-trained LBAE model...")
    if model is None:
        model = load_trained_model(device=device)
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    
    print("Step 2: Loading MNIST dataset...")
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    print("\nStep 3: Creating latent MNIST dataset...")
    train_latents, train_labels, test_latents, test_labels = process_dataset(
        model, train_loader, test_loader, device, output_dir)
    
    # Save model configuration for reference
    save_model_config(model, filepath=f"{output_dir}/model_config.json")
    
    print("\nLatent MNIST dataset creation complete!")
    print(f"Files are stored in '{output_dir}' directory")
    print(f"- Train examples: {len(train_latents)}")
    print(f"- Test examples: {len(test_latents)}")
    print(f"- Latent dimension: {model.latent_dim}")
    
    return {
        "train_latents": train_latents,
        "train_labels": train_labels,
        "test_latents": test_latents,
        "test_labels": test_labels,
        "model": model
    }

if __name__ == "__main__":
    create_latent_mnist_dataset()
