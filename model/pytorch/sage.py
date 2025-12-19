import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LSTMAggregator(nn.Module):
    """Aggregates neighbor features with an LSTM along the neighbor dimension.

    Args:
        input_dim: feature dimension of each neighbor.
        hidden_dim: output dimension of the aggregated embedding.
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(
            self,
            neighbor_feats: torch.Tensor,
            neighbor_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Aggregate neighbor features.

        Args:
            neighbor_feats: (batch, num_nodes, num_neighbors, input_dim)
            neighbor_weights: optional weights aligned with neighbors.
                Shape can be (num_nodes, num_neighbors) or (batch, num_nodes, num_neighbors[, 1]).
        Returns:
            agg: (batch, num_nodes, hidden_dim)
        """
        batch, num_nodes, num_neighbors, input_dim = neighbor_feats.shape
        feats = neighbor_feats
        if neighbor_weights is not None:
            if neighbor_weights.dim() == 2:
                weights = neighbor_weights.unsqueeze(0).expand(batch, -1, -1)
            else:
                weights = neighbor_weights
            if weights.dim() == 3:
                weights = weights.unsqueeze(-1)
            feats = feats * weights

        lstm_in = feats.reshape(batch * num_nodes, num_neighbors, input_dim)
        _, (agg, _) = self.lstm(lstm_in)
        agg = agg[-1].reshape(batch, num_nodes, -1)
        return agg


class GraphSAGEBlock(nn.Module):
    """GraphSAGE block using an LSTM aggregator."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_neighbors: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_neighbors = num_neighbors
        self.aggregator = LSTMAggregator(in_dim, out_dim)
        self.self_linear = nn.Linear(in_dim, out_dim)
        self.combine_linear = nn.Linear(out_dim * 2, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        neighbor_index: torch.Tensor,
        neighbor_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for a single time slice.

        Args:
            x: (batch, num_nodes, in_dim)
            neighbor_index: (num_nodes, num_neighbors)
            neighbor_weights: optional weights aligned with neighbor_index (num_nodes, num_neighbors)
        Returns:
            out: (batch, num_nodes, out_dim)
        """
        if neighbor_index.dim() != 2:
            raise ValueError("neighbor_index must be (num_nodes, num_neighbors)")

        batch, num_nodes, in_dim = x.shape
        neighbor_index = neighbor_index.to(x.device)
        neigh_feats = x[:, neighbor_index, :]  # (batch, num_nodes, num_neighbors, in_dim)

        agg = self.aggregator(neigh_feats, neighbor_weights)
        self_feats = self.self_linear(x)
        combined = torch.cat([self_feats, agg], dim=-1)
        out = self.combine_linear(combined)
        out = F.relu(out)
        return self.dropout(out)


class GraphSAGEEncoder(nn.Module):
    """Stack multiple GraphSAGE blocks."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_neighbors: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers = []
        dim_in = in_dim
        for _ in range(num_layers):
            layers.append(GraphSAGEBlock(dim_in, hidden_dim, num_neighbors, dropout=dropout))
            dim_in = hidden_dim
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        x: torch.Tensor,
        neighbor_index: torch.Tensor,
        neighbor_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h, neighbor_index, neighbor_weights)
        return h