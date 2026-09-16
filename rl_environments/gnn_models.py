import torch
import torch.nn as nn
import torch.nn.functional as F

import networkx as nx

from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.nn import GCNConv


class DynamicGraphPromptLayer(nn.Module):

    def __init__(self, context_dim, hidden_dim):
        super().__init__()
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        
        self.weight_generator = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh() 
        )
        
    def forward(self, node_embeddings: torch.Tensor, batch: torch.Tensor, context: torch.Tensor):


        generated_weights = self.weight_generator(context)
        weights_expanded = generated_weights[batch]
        weighted_embeddings = node_embeddings * weights_expanded
        graph_prompt = global_mean_pool(weighted_embeddings, batch=batch)
        
        return graph_prompt


class GNNActor(nn.Module):

    def __init__(self, input_dim, hidden_dim, action_shape):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.output_dim = int(action_shape.shape[0])
        self.additional_feature_vector_dim = 2
        self.prompt_layer = DynamicGraphPromptLayer(
            context_dim=self.additional_feature_vector_dim, 
            hidden_dim=hidden_dim
        )

        self.fc_mu = nn.Linear(hidden_dim + self.additional_feature_vector_dim, self.output_dim)
    
        self.sigma_param = nn.Parameter(torch.zeros(self.output_dim, 1))

    def forward(self, observations, state=None, info={}):
        batch_size = observations["graph_nodes"].shape[0]
        # PyG 
        data_list = [
            Data(
                x=observations["graph_nodes"][idx],
                edge_index=observations.graph_edge_links[0][:, observations.graph_edges[0]],
            )
            for idx in range(batch_size)
        ]
        batch_data = Batch.from_data_list(data_list)
        
        additional_features = observations["additional_features"].clone().detach()

        device = next(self.parameters()).device
        batch_data = batch_data.to(device)
        additional_features = additional_features.to(device)

        x = batch_data.x.float()
        edge_index = batch_data.edge_index.long()
        batch = batch_data.batch

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        prompt_graph_repr = self.prompt_layer(x, batch=batch, context=additional_features)

        final_repr = torch.cat([prompt_graph_repr, additional_features], dim=1)
        mu = self.fc_mu(final_repr)

        shape = [1] * len(mu.shape)
        shape[1] = -1
        sigma = (self.sigma_param.view(shape) + torch.zeros_like(mu)).exp()

        return (mu, sigma), state


class GNNCritic(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.additional_feature_vector_dim = 2

        self.prompt_layer = DynamicGraphPromptLayer(
            context_dim=self.additional_feature_vector_dim, 
            hidden_dim=hidden_dim
        )
        
        self.fc = nn.Linear(hidden_dim + self.additional_feature_vector_dim, 1)

    def forward(self, observations: nx.Graph, state=None, info={}):
        batch_size = observations["graph_nodes"].shape[0]

        data_list = [
            Data(
                x=observations["graph_nodes"][idx],
                edge_index=observations.graph_edge_links[0][:, observations.graph_edges[0]],
            )
            for idx in range(batch_size)
        ]
        batch_data = Batch.from_data_list(data_list)
        additional_features = observations["additional_features"].clone().detach()

        device = next(self.parameters()).device
        batch_data = batch_data.to(device)
        additional_features = additional_features.to(device)

        x = batch_data.x.float()
        edge_index = batch_data.edge_index.long()
        batch = batch_data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        prompt_graph_repr = self.prompt_layer(x, batch=batch, context=additional_features)

        final_repr = torch.cat([prompt_graph_repr, additional_features], dim=1)
        value = self.fc(final_repr)

        return value