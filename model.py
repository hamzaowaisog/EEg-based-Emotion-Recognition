import torch
import torch.nn as nn
import torch.nn.functional as F

class CLISAEncoder (nn.Module):
    ## Encoder for extracting spatiotemporal features.
    
    def __init__(self, num_channels, num_spatial_filters,temporal_filter_size):
        super(CLISAEncoder, self).__init__()
        self.spatial_conv = nn.Conv1d(
            in_channels=num_channels,
            out_channels=num_spatial_filters,
            kernel_size=1
        )
        self.temporal_conv = nn.Conv1d(
            in_channels=num_spatial_filters,
            out_channels=num_spatial_filters,
            kernel_size=temporal_filter_size
        )
        self.activation = nn.ReLU()
        
    def forward(self, x):
        x = self.spatial_conv(x)
        x = self.activation(x)
        x = self.temporal_conv(x)
        x = self.activation(x)
        return x
    
class Projector(nn.Module):
    ## Projector for mapping features to a latent space
    
    def __init__(self,input_dim, latent_dim):
        super(Projector, self).__init__()
        self.linear1 = nn.Linear(input_dim, latent_dim)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(latent_dim, latent_dim)
        
        
    def forward (self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x
    
    
def contrastive_loss(z_i , z_j , temperature=0.5):
    #Contrastive loss using cosine similarity
    sim = F.cosine_similarity(z_i, z_j)
    exp_sim = torch.exp(sim/temperature)
    return -torch.log(exp_sim / exp_sim.sum()).mean()
