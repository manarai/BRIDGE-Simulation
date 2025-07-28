"""
BRIDGE-VAE: Variational Autoencoder for Perturbation Simulation
Implementation of conditional VAE for predicting unseen perturbation responses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class BRIDGEDataset(Dataset):
    """
    Dataset class for BRIDGE multi-omic perturbation data
    """
    
    def __init__(self, 
                 regulon_activities: pd.DataFrame,
                 protein_activities: pd.DataFrame,
                 velocity_profiles: pd.DataFrame,
                 perturbation_descriptors: pd.DataFrame,
                 baseline_states: pd.DataFrame = None):
        """
        Initialize dataset with BRIDGE outputs and perturbation descriptors
        
        Parameters:
        -----------
        regulon_activities : pd.DataFrame
            Regulon activity matrix (cells x regulons)
        protein_activities : pd.DataFrame
            Protein activity matrix (cells x proteins)
        velocity_profiles : pd.DataFrame
            RNA velocity matrix (cells x genes)
        perturbation_descriptors : pd.DataFrame
            Perturbation characteristics (cells x descriptors)
        baseline_states : pd.DataFrame, optional
            Baseline cellular states before perturbation
        """
        self.regulon_activities = torch.FloatTensor(regulon_activities.values)
        self.protein_activities = torch.FloatTensor(protein_activities.values)
        self.velocity_profiles = torch.FloatTensor(velocity_profiles.values)
        self.perturbation_descriptors = torch.FloatTensor(perturbation_descriptors.values)
        
        if baseline_states is not None:
            self.baseline_states = torch.FloatTensor(baseline_states.values)
        else:
            # Use zeros as baseline if not provided
            self.baseline_states = torch.zeros_like(self.regulon_activities)
            
        # Concatenate all cellular state features
        self.cellular_states = torch.cat([
            self.regulon_activities,
            self.protein_activities,
            self.velocity_profiles
        ], dim=1)
        
        self.n_samples = len(self.cellular_states)
        
    def __len__(self):
        return self.n_samples
        
    def __getitem__(self, idx):
        return {
            'cellular_state': self.cellular_states[idx],
            'perturbation': self.perturbation_descriptors[idx],
            'baseline_state': self.baseline_states[idx],
            'regulon_activities': self.regulon_activities[idx],
            'protein_activities': self.protein_activities[idx],
            'velocity_profiles': self.velocity_profiles[idx]
        }

class ConditionalVAE(nn.Module):
    """
    Conditional Variational Autoencoder for perturbation simulation
    """
    
    def __init__(self,
                 cellular_state_dim: int,
                 perturbation_dim: int,
                 latent_dim: int = 128,
                 hidden_dims: List[int] = [512, 256],
                 dropout_rate: float = 0.1):
        """
        Initialize Conditional VAE
        
        Parameters:
        -----------
        cellular_state_dim : int
            Dimension of cellular state features
        perturbation_dim : int
            Dimension of perturbation descriptors
        latent_dim : int
            Dimension of latent space
        hidden_dims : list
            Hidden layer dimensions
        dropout_rate : float
            Dropout rate for regularization
        """
        super(ConditionalVAE, self).__init__()
        
        self.cellular_state_dim = cellular_state_dim
        self.perturbation_dim = perturbation_dim
        self.latent_dim = latent_dim
        
        # Encoder for cellular states
        encoder_layers = []
        input_dim = cellular_state_dim + perturbation_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            input_dim = hidden_dim
            
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space parameters
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        decoder_layers = []
        input_dim = latent_dim + perturbation_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            input_dim = hidden_dim
            
        decoder_layers.append(nn.Linear(input_dim, cellular_state_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, x, condition):
        """
        Encode input to latent parameters
        """
        x_cond = torch.cat([x, condition], dim=1)
        h = self.encoder(x_cond)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
        
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick for sampling
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z, condition):
        """
        Decode latent representation to output
        """
        z_cond = torch.cat([z, condition], dim=1)
        return self.decoder(z_cond)
        
    def forward(self, x, condition):
        """
        Forward pass through VAE
        """
        mu, logvar = self.encode(x, condition)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, condition)
        return recon_x, mu, logvar
        
    def sample(self, condition, n_samples=1):
        """
        Generate samples for given conditions
        """
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim)
            if condition.dim() == 1:
                condition = condition.unsqueeze(0).repeat(n_samples, 1)
            samples = self.decode(z, condition)
        return samples

class BRIDGEVAETrainer:
    """
    Trainer class for BRIDGE VAE
    """
    
    def __init__(self,
                 model: ConditionalVAE,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 learning_rate: float = 1e-3,
                 beta: float = 1.0):
        """
        Initialize trainer
        
        Parameters:
        -----------
        model : ConditionalVAE
            VAE model to train
        device : str
            Device for training
        learning_rate : float
            Learning rate for optimizer
        beta : float
            Beta parameter for beta-VAE
        """
        self.model = model.to(device)
        self.device = device
        self.beta = beta
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )
        
    def vae_loss(self, recon_x, x, mu, logvar):
        """
        Compute VAE loss (reconstruction + KL divergence)
        """
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + self.beta * kl_loss, recon_loss, kl_loss
        
    def train_epoch(self, dataloader):
        """
        Train for one epoch
        """
        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        
        for batch in dataloader:
            cellular_state = batch['cellular_state'].to(self.device)
            perturbation = batch['perturbation'].to(self.device)
            
            self.optimizer.zero_grad()
            
            recon_x, mu, logvar = self.model(cellular_state, perturbation)
            loss, recon_loss, kl_loss = self.vae_loss(recon_x, cellular_state, mu, logvar)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            
        n_samples = len(dataloader.dataset)
        return {
            'total_loss': total_loss / n_samples,
            'recon_loss': total_recon_loss / n_samples,
            'kl_loss': total_kl_loss / n_samples
        }
        
    def validate(self, dataloader):
        """
        Validate model
        """
        self.model.eval()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                cellular_state = batch['cellular_state'].to(self.device)
                perturbation = batch['perturbation'].to(self.device)
                
                recon_x, mu, logvar = self.model(cellular_state, perturbation)
                loss, recon_loss, kl_loss = self.vae_loss(recon_x, cellular_state, mu, logvar)
                
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
                
        n_samples = len(dataloader.dataset)
        return {
            'total_loss': total_loss / n_samples,
            'recon_loss': total_recon_loss / n_samples,
            'kl_loss': total_kl_loss / n_samples
        }
        
    def train(self, train_loader, val_loader, n_epochs=100, verbose=True):
        """
        Train the model
        """
        train_losses = []
        val_losses = []
        
        for epoch in range(n_epochs):
            # Training
            train_metrics = self.train_epoch(train_loader)
            train_losses.append(train_metrics)
            
            # Validation
            val_metrics = self.validate(val_loader)
            val_losses.append(val_metrics)
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['total_loss'])
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{n_epochs}")
                print(f"Train Loss: {train_metrics['total_loss']:.4f} "
                      f"(Recon: {train_metrics['recon_loss']:.4f}, "
                      f"KL: {train_metrics['kl_loss']:.4f})")
                print(f"Val Loss: {val_metrics['total_loss']:.4f} "
                      f"(Recon: {val_metrics['recon_loss']:.4f}, "
                      f"KL: {val_metrics['kl_loss']:.4f})")
                print("-" * 50)
                
        return train_losses, val_losses

class PerturbationSimulator:
    """
    High-level interface for perturbation simulation using trained VAE
    """
    
    def __init__(self, 
                 trained_model: ConditionalVAE,
                 regulon_dim: int,
                 protein_dim: int,
                 velocity_dim: int,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize simulator with trained model
        """
        self.model = trained_model.to(device)
        self.device = device
        self.regulon_dim = regulon_dim
        self.protein_dim = protein_dim
        self.velocity_dim = velocity_dim
        
    def simulate_perturbation(self, 
                            perturbation_descriptors: np.ndarray,
                            n_samples: int = 100,
                            return_uncertainty: bool = True) -> Dict:
        """
        Simulate cellular responses to perturbations
        
        Parameters:
        -----------
        perturbation_descriptors : np.ndarray
            Perturbation characteristics
        n_samples : int
            Number of samples to generate per perturbation
        return_uncertainty : bool
            Whether to return uncertainty estimates
            
        Returns:
        --------
        dict : Simulation results with mean predictions and uncertainties
        """
        self.model.eval()
        
        perturbation_tensor = torch.FloatTensor(perturbation_descriptors).to(self.device)
        
        if perturbation_tensor.dim() == 1:
            perturbation_tensor = perturbation_tensor.unsqueeze(0)
            
        n_perturbations = perturbation_tensor.shape[0]
        
        # Generate samples
        all_samples = []
        for i in range(n_perturbations):
            samples = self.model.sample(perturbation_tensor[i], n_samples)
            all_samples.append(samples.cpu().numpy())
            
        all_samples = np.array(all_samples)  # Shape: (n_perturbations, n_samples, feature_dim)
        
        # Split into different modalities
        regulon_samples = all_samples[:, :, :self.regulon_dim]
        protein_samples = all_samples[:, :, self.regulon_dim:self.regulon_dim + self.protein_dim]
        velocity_samples = all_samples[:, :, self.regulon_dim + self.protein_dim:]
        
        results = {
            'regulon_activities': {
                'mean': np.mean(regulon_samples, axis=1),
                'std': np.std(regulon_samples, axis=1) if return_uncertainty else None,
                'samples': regulon_samples if return_uncertainty else None
            },
            'protein_activities': {
                'mean': np.mean(protein_samples, axis=1),
                'std': np.std(protein_samples, axis=1) if return_uncertainty else None,
                'samples': protein_samples if return_uncertainty else None
            },
            'velocity_profiles': {
                'mean': np.mean(velocity_samples, axis=1),
                'std': np.std(velocity_samples, axis=1) if return_uncertainty else None,
                'samples': velocity_samples if return_uncertainty else None
            }
        }
        
        return results
        
    def interpolate_perturbations(self, 
                                perturbation_a: np.ndarray,
                                perturbation_b: np.ndarray,
                                n_steps: int = 10) -> Dict:
        """
        Interpolate between two perturbations to explore perturbation space
        """
        # Create interpolation path
        alphas = np.linspace(0, 1, n_steps)
        interpolated_perturbations = []
        
        for alpha in alphas:
            interp_pert = (1 - alpha) * perturbation_a + alpha * perturbation_b
            interpolated_perturbations.append(interp_pert)
            
        interpolated_perturbations = np.array(interpolated_perturbations)
        
        # Simulate responses along interpolation path
        results = self.simulate_perturbation(interpolated_perturbations, n_samples=50)
        results['interpolation_alphas'] = alphas
        
        return results

# Example usage and demonstration
def create_example_data():
    """
    Create example data for demonstration
    """
    n_cells = 1000
    n_regulons = 100
    n_proteins = 200
    n_velocity_genes = 500
    n_perturbation_features = 50
    
    # Simulate BRIDGE outputs
    regulon_activities = pd.DataFrame(
        np.random.beta(2, 5, size=(n_cells, n_regulons)),
        columns=[f'Regulon_{i}' for i in range(n_regulons)]
    )
    
    protein_activities = pd.DataFrame(
        np.random.gamma(2, 0.5, size=(n_cells, n_proteins)),
        columns=[f'Protein_{i}' for i in range(n_proteins)]
    )
    
    velocity_profiles = pd.DataFrame(
        np.random.normal(0, 0.5, size=(n_cells, n_velocity_genes)),
        columns=[f'Gene_{i}' for i in range(n_velocity_genes)]
    )
    
    # Simulate perturbation descriptors (e.g., chemical fingerprints)
    perturbation_descriptors = pd.DataFrame(
        np.random.uniform(0, 1, size=(n_cells, n_perturbation_features)),
        columns=[f'Feature_{i}' for i in range(n_perturbation_features)]
    )
    
    return regulon_activities, protein_activities, velocity_profiles, perturbation_descriptors

if __name__ == "__main__":
    # Example usage
    print("Creating example data...")
    regulon_act, protein_act, velocity_prof, pert_desc = create_example_data()
    
    # Create dataset
    dataset = BRIDGEDataset(regulon_act, protein_act, velocity_prof, pert_desc)
    
    # Split into train/validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    cellular_state_dim = regulon_act.shape[1] + protein_act.shape[1] + velocity_prof.shape[1]
    perturbation_dim = pert_desc.shape[1]
    
    model = ConditionalVAE(
        cellular_state_dim=cellular_state_dim,
        perturbation_dim=perturbation_dim,
        latent_dim=64,
        hidden_dims=[256, 128]
    )
    
    # Train model
    trainer = BRIDGEVAETrainer(model, beta=0.5)
    print("Training VAE...")
    train_losses, val_losses = trainer.train(train_loader, val_loader, n_epochs=50)
    
    # Create simulator
    simulator = PerturbationSimulator(
        model, 
        regulon_act.shape[1], 
        protein_act.shape[1], 
        velocity_prof.shape[1]
    )
    
    # Simulate novel perturbation
    novel_perturbation = np.random.uniform(0, 1, size=(1, perturbation_dim))
    results = simulator.simulate_perturbation(novel_perturbation, n_samples=100)
    
    print("Simulation complete!")
    print(f"Predicted regulon activities shape: {results['regulon_activities']['mean'].shape}")
    print(f"Predicted protein activities shape: {results['protein_activities']['mean'].shape}")
    print(f"Predicted velocity profiles shape: {results['velocity_profiles']['mean'].shape}")

