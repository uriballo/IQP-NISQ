import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import schedulefree
from utils.visualizations import visualize_results

def train_lbae(model, train_loader, optimizer, device, epoch, log_interval=100):
    model.train()
    optimizer.train()
    train_loss = 0
    
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), 
                        desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, (data, _) in progress_bar:
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        recon_batch, binary_latent, _ = model(data)
        
        # BCE loss for MNIST (as mentioned in the paper)
        loss = F.binary_cross_entropy(recon_batch, data, reduction='sum')
            
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix(loss=loss.item() / len(data))
    
    avg_loss = train_loss / len(train_loader.dataset)
    print(f'====> Epoch: {epoch} Average train loss: {avg_loss:.4f}')
    
    return avg_loss

def test_lbae(model, test_loader, device):
    model.eval()
    test_loss = 0
    
    progress_bar = tqdm(test_loader, desc="Testing")
    
    with torch.no_grad():
        for data, _ in progress_bar:
            data = data.to(device)
            
            # Forward pass
            recon_batch, binary_latent, _ = model(data)
            
            # BCE loss for MNIST
            loss = F.binary_cross_entropy(recon_batch, data, reduction='sum')
                
            test_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix(loss=loss.item() / len(data))
    
    avg_loss = test_loss / len(test_loader.dataset)
    print(f'====> Test set loss: {avg_loss:.4f}')
    
    return avg_loss

def train_model(model, train_loader, test_loader, device, num_epochs=20, lr=1e-3):
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=lr)
    
    train_losses = []
    test_losses = []
    
    # Create output directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    best_test_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        train_loss = train_lbae(model, train_loader, optimizer, device, epoch)
        optimizer.eval()
        test_loss = test_lbae(model, test_loader, device)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        # Save best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'test_loss': test_loss,
            }, 'models/lbae_best.pt')
        
        # Save checkpoint
        if epoch % 10 == 0 or epoch == num_epochs:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'test_loss': test_loss,
            }, f'models/lbae_epoch_{epoch}.pt')
        
        # Plot progress every 10 epochs
        if epoch % 10 == 0 or epoch == num_epochs:
            # Plot training/test loss
            plt.figure(figsize=(10, 5))
            plt.plot(train_losses, label='Train Loss')
            plt.plot(test_losses, label='Test Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('LBAE Training and Test Loss')
            plt.savefig('plots/lbae_loss.png')
            plt.close()
            
            # Visualize current results
            visualize_results(model, test_loader, device)
    
    return model, train_losses, test_losses
