import os
import numpy as np
import torch
from tqdm import tqdm
import json
from utils.visualizations import plot_latent_mnist_samples

def create_latent_dataset(model, data_loader, device, output_dir, split_name):
    """Convert dataset to latent representation and save as numpy arrays"""
    model.eval()
    
    all_latents = []
    all_labels = []
    
    print(f"Converting {split_name} dataset to latent representation...")
    with torch.no_grad():
        for data, labels in tqdm(data_loader):
            data = data.to(device)
            binary_latent, _ = model.encode(data)
            
            # Convert from {-1, 1} to {0, 1} for storage
            binary_latent = (binary_latent + 1) / 2
            
            all_latents.append(binary_latent.cpu().numpy())
            all_labels.append(labels.numpy())
    
    # Concatenate all batches
    latents = np.concatenate(all_latents, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as numpy files
    np.save(f"{output_dir}/{split_name}_latents.npy", latents.astype(np.float32))
    np.save(f"{output_dir}/{split_name}_labels.npy", labels)
    
    print(f"Saved {len(latents)} {split_name} samples to {output_dir}")
    
    # Save metadata
    metadata_path = f"{output_dir}/metadata.json"
    if not os.path.exists(metadata_path):
        metadata = {
            "latent_dim": model.latent_dim,
            "description": "MNIST dataset encoded to binary latent space using LBAE"
        }
    else:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    # Update metadata with split information
    if split_name == "train":
        metadata["num_train"] = len(latents)
    else:
        metadata["num_test"] = len(latents)
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)
    
    return latents, labels

def process_dataset(model, train_loader, test_loader, device, output_dir="latent_mnist_data"):
    """Process both train and test splits to create latent dataset"""
    # Create latent representations
    train_latents, train_labels = create_latent_dataset(
        model, train_loader, device, output_dir, "train")
    
    test_latents, test_labels = create_latent_dataset(
        model, test_loader, device, output_dir, "test")
    
    print("Latent dataset creation complete!")
    
    # Visualize some samples
    plot_latent_mnist_samples(
        train_latents, train_labels, model, device, 
        output_path=f"{output_dir}/train_samples.png")
    
    plot_latent_mnist_samples(
        test_latents, test_labels, model, device,
        output_path=f"{output_dir}/test_samples.png")
    
    return train_latents, train_labels, test_latents, test_labels
