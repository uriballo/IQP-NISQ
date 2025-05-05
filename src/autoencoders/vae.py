import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# --- Configuration ---
input_dim = 91
latent_dim = 21
intermediate_dim = 64
initial_temperature = 2.0  # Starting temperature for Gumbel-Softmax
min_temperature = 0.1  # Minimum temperature
anneal_rate = 0.0003  # Temperature annealing rate per step/batch


# --- Model Definition ---
class BinaryAutoencoderGumbel(nn.Module):
    def __init__(
        self, input_dim, latent_dim, intermediate_dim, temperature
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.temperature = temperature

        # Encoder layers
        self.enc_fc1 = nn.Linear(input_dim, intermediate_dim)
        self.enc_fc2_logits = nn.Linear(
            intermediate_dim, latent_dim * 2
        )  # Output 2 logits per latent dim

        # Decoder layers
        self.dec_fc1 = nn.Linear(latent_dim, intermediate_dim)
        self.dec_fc2_output = nn.Linear(intermediate_dim, input_dim)

    def encode(self, x, current_temp):
        """Encodes input to binary latent space using Gumbel-Softmax."""
        x = F.relu(self.enc_fc1(x))
        logits = self.enc_fc2_logits(x)

        # Reshape logits to (batch_size, latent_dim, 2)
        # Each pair [logit_0, logit_1] represents the log-probabilities
        # for the latent variable being 0 or 1 respectively.
        logits_reshaped = logits.view(-1, self.latent_dim, 2)

        # Apply Gumbel-Softmax
        # hard=True: returns one-hot vectors in forward pass,
        # but uses soft probabilities for gradient calculation.
        # tau: temperature parameter
        z_one_hot = F.gumbel_softmax(
            logits_reshaped, tau=current_temp, hard=True, dim=-1
        )

        # Convert one-hot representation to binary (0 or 1)
        # The second element [:, :, 1] corresponds to the probability/value of '1'
        z_binary = z_one_hot[:, :, 1]

        return z_binary, logits_reshaped # Return logits for potential KL divergence loss later if needed

    def decode(self, z):
        """Decodes latent space representation back to reconstruction."""
        x = F.relu(self.dec_fc1(z))
        reconstruction = torch.sigmoid(
            self.dec_fc2_output(x)
        )  # Sigmoid for BCE loss
        return reconstruction

    def forward(self, x, current_temp):
        """Full forward pass."""
        z_binary, _ = self.encode(x, current_temp)
        reconstruction = self.decode(z_binary)
        return reconstruction

    def update_temperature(self, current_step, anneal_rate, min_temp):
        """Anneal temperature exponentially."""
        self.temperature = np.maximum(
            initial_temperature * np.exp(-anneal_rate * current_step),
            min_temp,
        )
        return self.temperature


# --- Training Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = BinaryAutoencoderGumbel(
    input_dim, latent_dim, intermediate_dim, initial_temperature
).to(device)
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print("\nModel Architecture:")
print(model)

# --- Dummy Data Example ---
batch_size = 64
num_samples = 1024
# Generate random binary data (0.0 or 1.0) as float32
x_train_np = np.random.randint(
    0, 2, size=(num_samples, input_dim)
).astype(np.float32)
x_train = torch.from_numpy(x_train_np).to(device)

# --- Training Loop Example ---
epochs = 50
steps_per_epoch = len(x_train) // batch_size
total_steps = 0

print("\nStarting Training...")
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    current_temp = model.temperature # Get temp at start of epoch

    # Simple example without DataLoader for clarity
    permutation = torch.randperm(x_train.size(0))
    for i in range(0, x_train.size(0), batch_size):
        indices = permutation[i : i + batch_size]
        batch_x = x_train[indices]

        optimizer.zero_grad()

        # Update temperature for the current step
        current_temp = model.update_temperature(
            total_steps, anneal_rate, min_temperature
        )

        # Forward pass
        reconstructions = model(batch_x, current_temp)

        # Calculate loss
        loss = criterion(reconstructions, batch_x)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        total_steps += 1

    avg_epoch_loss = epoch_loss / steps_per_epoch
    print(
        f"Epoch [{epoch+1}/{epochs}], Loss: {avg_epoch_loss:.4f}, Temp: {current_temp:.4f}"
    )

print("Training Finished.")

# --- Inference Example ---
model.eval()
with torch.no_grad():
    # Get binary latent codes for the first 5 samples
    example_input = x_train[:5]
    # Use a low temperature for inference, or the final annealed temperature
    final_temp = model.temperature
    latent_codes_binary, _ = model.encode(example_input, final_temp)
    reconstructions_continuous = model.decode(latent_codes_binary)
    # Round reconstructions to get binary output if needed
    reconstructions_binary = torch.round(reconstructions_continuous)

    print("\n--- Inference Example ---")
    print("Original Input (first 5):")
    print(example_input.cpu().numpy().astype(int))
    print("\nBinary Latent Codes (first 5):")
    print(latent_codes_binary.cpu().numpy().astype(int))
    print("\nReconstructed Output (binary, first 5):")
    print(reconstructions_binary.cpu().numpy().astype(int))

