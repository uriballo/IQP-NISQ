import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Binary straight-through estimator for binarizing the latent space
class BinarizationLayer(nn.Module):
    def __init__(self):
        super(BinarizationLayer, self).__init__()
        
    def forward(self, z):
        # Forward: Binarize to {-1, 1}
        b = torch.sign(z)
        # Handle zero values (sign(0) = 0, we want 1)
        b = torch.where(b == 0, torch.ones_like(b), b)
        
        # Straight-through estimator for backward pass
        # Use z directly for gradient (unit gradient surrogate function)
        b = b.detach() + z - z.detach()
        
        return b

# Encoder Residual Block
class EncoderResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(EncoderResBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.act = nn.LeakyReLU(0.02)
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(residual)
        out = self.act(out)
        
        return out

# Decoder Residual Block
class DecoderResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DecoderResBlock, self).__init__()
        
        # Use transposed convolutions for upsampling when stride > 1
        if stride > 1:
            self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, 
                                          kernel_size=4, stride=stride, 
                                          padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                  stride=1, padding=1, bias=False)
                                  
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            if stride > 1:
                self.shortcut = nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels,
                                     kernel_size=4, stride=stride,
                                     padding=1, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1,
                             stride=1, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
                
        self.act = nn.LeakyReLU(0.02)
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(residual)
        out = self.act(out)
        
        return out

# Enhanced LBAE with Convolutional Architecture
class ImprovedLBAE(nn.Module):
    def __init__(self, input_channels=1, img_size=28, latent_dim=32):
        super(ImprovedLBAE, self).__init__()
        
        self.input_channels = input_channels
        self.img_size = img_size
        self.latent_dim = latent_dim
        
        # Base feature maps
        base_channels = 32
        
        # Encoding layers
        self.enc_initial = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.02)
        )
        
        # Residual blocks with downsampling
        self.enc_block1 = EncoderResBlock(base_channels, base_channels * 2, stride=2)  # Size/2
        self.enc_block2 = EncoderResBlock(base_channels * 2, base_channels * 4, stride=2)  # Size/4
        self.enc_block3 = EncoderResBlock(base_channels * 4, base_channels * 8, stride=2)  # Size/8
        self.enc_block4 = EncoderResBlock(base_channels * 8, base_channels * 8, stride=2)  # Size/16
        
        # Binarization layer
        self.binarization = BinarizationLayer()
        
        # Placeholder for the linear layer - will be set in first forward pass
        self.fc_z = None
        self.latent_to_features = None
        
        # Number of feature maps in the bottleneck
        bottleneck_channels = base_channels * 8
        self.bottleneck_channels = bottleneck_channels
        
        # Decoding layers
        self.dec_reshape = None  # Will be set in first forward pass
        
        # Residual blocks with upsampling
        self.dec_block1 = DecoderResBlock(bottleneck_channels, base_channels * 4, stride=2)  # Size*2
        self.dec_block2 = DecoderResBlock(base_channels * 4, base_channels * 2, stride=2)  # Size*4
        self.dec_block3 = DecoderResBlock(base_channels * 2, base_channels, stride=2)  # Size*8
        
        # Final layer to reconstruct image
        self.dec_final = nn.Sequential(
            nn.ConvTranspose2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.02),
            nn.Conv2d(base_channels, input_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Sigmoid()  # For MNIST (values between 0 and 1)
        )
        
        self.initialized = False
        
    def _initialize_fc_layers(self, h):
        """Initialize the fully connected layers based on feature map size"""
        # Get the flattened dimension from the actual tensor
        flattened_dim = h.size(1)
        
        # Initialize the FC layers
        self.fc_z = nn.Sequential(
            nn.Linear(flattened_dim, self.latent_dim),
            nn.Tanh()  # tanh for binarization as in the paper
        ).to(h.device)
        
        self.latent_to_features = nn.Sequential(
            nn.Linear(self.latent_dim, flattened_dim),
            nn.LeakyReLU(0.02)
        ).to(h.device)
        
        # Get the spatial dimensions
        batch_size = h.size(0)
        feature_shape = h.view(batch_size, self.bottleneck_channels, -1).size(2)
        feature_size = int(np.sqrt(feature_shape))
        
        self.dec_reshape = nn.Unflatten(1, (self.bottleneck_channels, feature_size, feature_size))
        
        self.initialized = True
        
    def encode(self, x):
        """Encode input to binary latent space"""
        # Initial convolution
        h = self.enc_initial(x)
        
        # Residual blocks
        h = self.enc_block1(h)
        h = self.enc_block2(h)
        h = self.enc_block3(h)
        h = self.enc_block4(h)
        
        # Flatten
        h_flat = h.view(h.size(0), -1)
        
        # Initialize FC layers if first pass
        if not self.initialized:
            self._initialize_fc_layers(h_flat)
        
        # Project to latent space
        z = self.fc_z(h_flat)
        
        # Apply binarization
        b = self.binarization(z)
        
        return b, z
    
    def decode(self, b):
        """Decode from binary latent space"""
        if not self.initialized:
            raise RuntimeError("Model has not been initialized with a first forward pass")
        
        # Project to feature space
        h = self.latent_to_features(b)
        
        # Reshape to spatial features
        h = self.dec_reshape(h)
        
        # Residual blocks
        h = self.dec_block1(h)
        h = self.dec_block2(h)
        h = self.dec_block3(h)
        
        # Final output
        x_recon = self.dec_final(h)
        
        return x_recon
    
    def forward(self, x):
        """Full forward pass: encode and decode"""
        b, z = self.encode(x)
        x_recon = self.decode(b)
        return x_recon, b, z

def save_model_config(model, filepath="models/lbae_config.json"):
    """Save model configuration for later reconstruction"""
    import json
    import os
    
    config = {
        "input_channels": model.input_channels,
        "img_size": model.img_size,
        "latent_dim": model.latent_dim,
    }
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(config, f)
    
    print(f"Model configuration saved to {filepath}")

def load_trained_model(weights_path="models/lbae_best.pt", config_path="models/lbae_config.json", device=None):
    """Load a trained LBAE model from saved weights and config"""
    import json
    import torch
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Initialize model
    model = ImprovedLBAE(
        input_channels=config["input_channels"],
        img_size=config["img_size"],
        latent_dim=config["latent_dim"]
    ).to(device)
    
    # Load weights
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Initialize the model with a dummy forward pass
    dummy_input = torch.zeros((1, config["input_channels"], config["img_size"], config["img_size"])).to(device)
    with torch.no_grad():
        model(dummy_input)
    
    model.eval()
    print(f"Model loaded from {weights_path} with latent dimension: {model.latent_dim}")
    return model
