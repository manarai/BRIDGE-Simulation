"""
BRIDGE-GAN: Generative Adversarial Network for Perturbation Simulation
Implementation of conditional GAN for predicting unseen perturbation responses
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

class Generator(nn.Module):
    """
    Generator network for conditional GAN
    """
    
    def __init__(self,
                 noise_dim: int,
                 perturbation_dim: int,
                 cellular_state_dim: int,
                 hidden_dims: List[int] = [256, 512, 256],
                 dropout_rate: float = 0.1):
        """
        Initialize Generator
        
        Parameters:
        -----------
        noise_dim : int
            Dimension of input noise vector
        perturbation_dim : int
            Dimension of perturbation descriptors
        cellular_state_dim : int
            Dimension of output cellular state
        hidden_dims : list
            Hidden layer dimensions
        dropout_rate : float
            Dropout rate for regularization
        """
        super(Generator, self).__init__()
        
        self.noise_dim = noise_dim
        self.perturbation_dim = perturbation_dim
        self.cellular_state_dim = cellular_state_dim
        
        # Build generator network
        layers = []
        input_dim = noise_dim + perturbation_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate)
            ])
            input_dim = hidden_dim
            
        # Output layer
        layers.append(nn.Linear(input_dim, cellular_state_dim))
        layers.append(nn.Tanh())  # Normalize outputs to [-1, 1]
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, noise, perturbation):
        """
        Forward pass through generator
        """
        x = torch.cat([noise, perturbation], dim=1)
        return self.network(x)

class Discriminator(nn.Module):
    """
    Discriminator network for conditional GAN
    """
    
    def __init__(self,
                 cellular_state_dim: int,
                 perturbation_dim: int,
                 hidden_dims: List[int] = [512, 256, 128],
                 dropout_rate: float = 0.3):
        """
        Initialize Discriminator
        
        Parameters:
        -----------
        cellular_state_dim : int
            Dimension of cellular state input
        perturbation_dim : int
            Dimension of perturbation descriptors
        hidden_dims : list
            Hidden layer dimensions
        dropout_rate : float
            Dropout rate for regularization
        """
        super(Discriminator, self).__init__()
        
        self.cellular_state_dim = cellular_state_dim
        self.perturbation_dim = perturbation_dim
        
        # Build discriminator network
        layers = []
        input_dim = cellular_state_dim + perturbation_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate)
            ])
            input_dim = hidden_dim
            
        # Output layer (binary classification)
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, cellular_state, perturbation):
        """
        Forward pass through discriminator
        """
        x = torch.cat([cellular_state, perturbation], dim=1)
        return self.network(x)

class SpectralNorm(nn.Module):
    """
    Spectral normalization for improved GAN training stability
    """
    
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = F.normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = F.normalize(torch.mv(w.view(height,-1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = F.normalize(u.data)
        v.data = F.normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

class BRIDGEGANTrainer:
    """
    Trainer class for BRIDGE conditional GAN
    """
    
    def __init__(self,
                 generator: Generator,
                 discriminator: Discriminator,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 lr_g: float = 2e-4,
                 lr_d: float = 2e-4,
                 beta1: float = 0.5,
                 beta2: float = 0.999,
                 lambda_gp: float = 10.0):
        """
        Initialize GAN trainer
        
        Parameters:
        -----------
        generator : Generator
            Generator network
        discriminator : Discriminator
            Discriminator network
        device : str
            Device for training
        lr_g : float
            Learning rate for generator
        lr_d : float
            Learning rate for discriminator
        beta1, beta2 : float
            Adam optimizer parameters
        lambda_gp : float
            Gradient penalty coefficient
        """
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.device = device
        self.lambda_gp = lambda_gp
        
        # Optimizers
        self.optimizer_g = optim.Adam(
            self.generator.parameters(), 
            lr=lr_g, 
            betas=(beta1, beta2)
        )
        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(), 
            lr=lr_d, 
            betas=(beta1, beta2)
        )
        
        # Loss function
        self.criterion = nn.BCELoss()
        
    def gradient_penalty(self, real_samples, fake_samples, perturbations):
        """
        Compute gradient penalty for WGAN-GP
        """
        batch_size = real_samples.size(0)
        alpha = torch.rand(batch_size, 1).to(self.device)
        alpha = alpha.expand_as(real_samples)
        
        interpolated = alpha * real_samples + (1 - alpha) * fake_samples
        interpolated = interpolated.requires_grad_(True)
        
        d_interpolated = self.discriminator(interpolated, perturbations)
        
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty
        
    def train_discriminator(self, real_data, perturbations):
        """
        Train discriminator for one step
        """
        batch_size = real_data.size(0)
        
        # Train with real data
        self.optimizer_d.zero_grad()
        
        real_labels = torch.ones(batch_size, 1).to(self.device)
        real_output = self.discriminator(real_data, perturbations)
        d_loss_real = self.criterion(real_output, real_labels)
        
        # Train with fake data
        noise = torch.randn(batch_size, self.generator.noise_dim).to(self.device)
        fake_data = self.generator(noise, perturbations)
        fake_labels = torch.zeros(batch_size, 1).to(self.device)
        fake_output = self.discriminator(fake_data.detach(), perturbations)
        d_loss_fake = self.criterion(fake_output, fake_labels)
        
        # Gradient penalty
        gp = self.gradient_penalty(real_data, fake_data, perturbations)
        
        # Total discriminator loss
        d_loss = d_loss_real + d_loss_fake + self.lambda_gp * gp
        d_loss.backward()
        self.optimizer_d.step()
        
        return {
            'loss': d_loss.item(),
            'loss_real': d_loss_real.item(),
            'loss_fake': d_loss_fake.item(),
            'gradient_penalty': gp.item()
        }
        
    def train_generator(self, perturbations):
        """
        Train generator for one step
        """
        batch_size = perturbations.size(0)
        
        self.optimizer_g.zero_grad()
        
        # Generate fake data
        noise = torch.randn(batch_size, self.generator.noise_dim).to(self.device)
        fake_data = self.generator(noise, perturbations)
        
        # Try to fool discriminator
        real_labels = torch.ones(batch_size, 1).to(self.device)
        fake_output = self.discriminator(fake_data, perturbations)
        g_loss = self.criterion(fake_output, real_labels)
        
        g_loss.backward()
        self.optimizer_g.step()
        
        return {'loss': g_loss.item()}
        
    def train_epoch(self, dataloader, n_critic=5):
        """
        Train for one epoch
        """
        self.generator.train()
        self.discriminator.train()
        
        d_losses = []
        g_losses = []
        
        for i, batch in enumerate(dataloader):
            cellular_state = batch['cellular_state'].to(self.device)
            perturbation = batch['perturbation'].to(self.device)
            
            # Train discriminator
            d_metrics = self.train_discriminator(cellular_state, perturbation)
            d_losses.append(d_metrics)
            
            # Train generator (less frequently)
            if i % n_critic == 0:
                g_metrics = self.train_generator(perturbation)
                g_losses.append(g_metrics)
                
        return {
            'discriminator': {
                'loss': np.mean([d['loss'] for d in d_losses]),
                'loss_real': np.mean([d['loss_real'] for d in d_losses]),
                'loss_fake': np.mean([d['loss_fake'] for d in d_losses]),
                'gradient_penalty': np.mean([d['gradient_penalty'] for d in d_losses])
            },
            'generator': {
                'loss': np.mean([g['loss'] for g in g_losses])
            }
        }
        
    def train(self, dataloader, n_epochs=100, verbose=True):
        """
        Train the GAN
        """
        train_history = []
        
        for epoch in range(n_epochs):
            metrics = self.train_epoch(dataloader)
            train_history.append(metrics)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{n_epochs}")
                print(f"D Loss: {metrics['discriminator']['loss']:.4f} "
                      f"(Real: {metrics['discriminator']['loss_real']:.4f}, "
                      f"Fake: {metrics['discriminator']['loss_fake']:.4f}, "
                      f"GP: {metrics['discriminator']['gradient_penalty']:.4f})")
                print(f"G Loss: {metrics['generator']['loss']:.4f}")
                print("-" * 50)
                
        return train_history

class GANPerturbationSimulator:
    """
    High-level interface for perturbation simulation using trained GAN
    """
    
    def __init__(self, 
                 trained_generator: Generator,
                 regulon_dim: int,
                 protein_dim: int,
                 velocity_dim: int,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize simulator with trained generator
        """
        self.generator = trained_generator.to(device)
        self.device = device
        self.regulon_dim = regulon_dim
        self.protein_dim = protein_dim
        self.velocity_dim = velocity_dim
        
    def simulate_perturbation(self, 
                            perturbation_descriptors: np.ndarray,
                            n_samples: int = 100) -> Dict:
        """
        Simulate cellular responses to perturbations using GAN
        
        Parameters:
        -----------
        perturbation_descriptors : np.ndarray
            Perturbation characteristics
        n_samples : int
            Number of samples to generate per perturbation
            
        Returns:
        --------
        dict : Simulation results
        """
        self.generator.eval()
        
        perturbation_tensor = torch.FloatTensor(perturbation_descriptors).to(self.device)
        
        if perturbation_tensor.dim() == 1:
            perturbation_tensor = perturbation_tensor.unsqueeze(0)
            
        n_perturbations = perturbation_tensor.shape[0]
        
        # Generate samples
        all_samples = []
        with torch.no_grad():
            for i in range(n_perturbations):
                # Generate multiple samples for this perturbation
                noise = torch.randn(n_samples, self.generator.noise_dim).to(self.device)
                perturbation_repeated = perturbation_tensor[i].unsqueeze(0).repeat(n_samples, 1)
                
                samples = self.generator(noise, perturbation_repeated)
                # Convert from [-1, 1] to appropriate range
                samples = (samples + 1) / 2  # Convert to [0, 1]
                all_samples.append(samples.cpu().numpy())
                
        all_samples = np.array(all_samples)  # Shape: (n_perturbations, n_samples, feature_dim)
        
        # Split into different modalities
        regulon_samples = all_samples[:, :, :self.regulon_dim]
        protein_samples = all_samples[:, :, self.regulon_dim:self.regulon_dim + self.protein_dim]
        velocity_samples = all_samples[:, :, self.regulon_dim + self.protein_dim:]
        
        results = {
            'regulon_activities': {
                'mean': np.mean(regulon_samples, axis=1),
                'std': np.std(regulon_samples, axis=1),
                'samples': regulon_samples
            },
            'protein_activities': {
                'mean': np.mean(protein_samples, axis=1),
                'std': np.std(protein_samples, axis=1),
                'samples': protein_samples
            },
            'velocity_profiles': {
                'mean': np.mean(velocity_samples, axis=1),
                'std': np.std(velocity_samples, axis=1),
                'samples': velocity_samples
            }
        }
        
        return results
        
    def generate_diverse_responses(self, 
                                 perturbation_descriptor: np.ndarray,
                                 n_samples: int = 1000,
                                 diversity_threshold: float = 0.1) -> Dict:
        """
        Generate diverse responses to explore the space of possible outcomes
        """
        self.generator.eval()
        
        perturbation_tensor = torch.FloatTensor(perturbation_descriptor).to(self.device)
        if perturbation_tensor.dim() == 1:
            perturbation_tensor = perturbation_tensor.unsqueeze(0)
            
        # Generate many samples
        all_samples = []
        with torch.no_grad():
            for _ in range(n_samples // 100):  # Generate in batches
                noise = torch.randn(100, self.generator.noise_dim).to(self.device)
                perturbation_repeated = perturbation_tensor.repeat(100, 1)
                
                samples = self.generator(noise, perturbation_repeated)
                samples = (samples + 1) / 2  # Convert to [0, 1]
                all_samples.append(samples.cpu().numpy())
                
        all_samples = np.concatenate(all_samples, axis=0)
        
        # Cluster samples to identify diverse response modes
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        
        # Find optimal number of clusters
        best_k = 2
        best_score = -1
        for k in range(2, min(20, n_samples // 50)):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(all_samples)
            score = silhouette_score(all_samples, labels)
            if score > best_score:
                best_score = score
                best_k = k
                
        # Final clustering
        kmeans = KMeans(n_clusters=best_k, random_state=42)
        labels = kmeans.fit_predict(all_samples)
        
        # Extract representative samples from each cluster
        diverse_responses = []
        for cluster_id in range(best_k):
            cluster_samples = all_samples[labels == cluster_id]
            cluster_center = np.mean(cluster_samples, axis=0)
            diverse_responses.append(cluster_center)
            
        diverse_responses = np.array(diverse_responses)
        
        # Split into modalities
        regulon_responses = diverse_responses[:, :self.regulon_dim]
        protein_responses = diverse_responses[:, self.regulon_dim:self.regulon_dim + self.protein_dim]
        velocity_responses = diverse_responses[:, self.regulon_dim + self.protein_dim:]
        
        return {
            'diverse_responses': {
                'regulon_activities': regulon_responses,
                'protein_activities': protein_responses,
                'velocity_profiles': velocity_responses,
                'cluster_labels': labels,
                'n_clusters': best_k
            },
            'all_samples': {
                'regulon_activities': all_samples[:, :self.regulon_dim],
                'protein_activities': all_samples[:, self.regulon_dim:self.regulon_dim + self.protein_dim],
                'velocity_profiles': all_samples[:, self.regulon_dim + self.protein_dim:]
            }
        }

# Example usage
if __name__ == "__main__":
    # Import the dataset from VAE implementation
    from bridge_vae_implementation import BRIDGEDataset, create_example_data
    
    print("Creating example data...")
    regulon_act, protein_act, velocity_prof, pert_desc = create_example_data()
    
    # Create dataset
    dataset = BRIDGEDataset(regulon_act, protein_act, velocity_prof, pert_desc)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize GAN
    noise_dim = 100
    cellular_state_dim = regulon_act.shape[1] + protein_act.shape[1] + velocity_prof.shape[1]
    perturbation_dim = pert_desc.shape[1]
    
    generator = Generator(
        noise_dim=noise_dim,
        perturbation_dim=perturbation_dim,
        cellular_state_dim=cellular_state_dim,
        hidden_dims=[256, 512, 256]
    )
    
    discriminator = Discriminator(
        cellular_state_dim=cellular_state_dim,
        perturbation_dim=perturbation_dim,
        hidden_dims=[512, 256, 128]
    )
    
    # Train GAN
    trainer = BRIDGEGANTrainer(generator, discriminator)
    print("Training GAN...")
    train_history = trainer.train(dataloader, n_epochs=50)
    
    # Create simulator
    simulator = GANPerturbationSimulator(
        generator, 
        regulon_act.shape[1], 
        protein_act.shape[1], 
        velocity_prof.shape[1]
    )
    
    # Simulate novel perturbation
    novel_perturbation = np.random.uniform(0, 1, size=(perturbation_dim,))
    results = simulator.simulate_perturbation(novel_perturbation, n_samples=100)
    
    print("GAN simulation complete!")
    print(f"Predicted regulon activities shape: {results['regulon_activities']['mean'].shape}")
    print(f"Predicted protein activities shape: {results['protein_activities']['mean'].shape}")
    print(f"Predicted velocity profiles shape: {results['velocity_profiles']['mean'].shape}")
    
    # Generate diverse responses
    diverse_results = simulator.generate_diverse_responses(novel_perturbation, n_samples=500)
    print(f"Found {diverse_results['diverse_responses']['n_clusters']} diverse response modes")

