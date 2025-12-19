import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from model.pytorch.sage import GraphSAGEEncoder
from model.pytorch.tcn import TemporalConvNet


def _init_neighbor_index(num_nodes: int, num_neighbors: int, device: torch.device) -> torch.Tensor:
    num_neighbors = min(num_neighbors, num_nodes)
    index = torch.randint(num_nodes, (num_nodes, num_neighbors), device=device)
    return index


def _as_int_list(maybe_list, fallback: List[int]) -> List[int]:
    if maybe_list is None:
        return fallback
    if isinstance(maybe_list, int):
        return [maybe_list]
    if isinstance(maybe_list, (list, tuple)):
        return [int(v) for v in maybe_list]
    return fallback


class NodeEmbeddingMLP(nn.Module):
    """Small MLP for static node features.

    Keeps the naming fc3/fc4/fc5 to mirror legacy SAGDFN implementations.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.fc3 = nn.Linear(in_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, out_dim)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (num_nodes, node_feat_dim)
        h = self.act(self.fc3(x))
        h = self.act(self.fc4(h))
        h = self.fc5(h)
        return h


class SAGDFNModel(nn.Module):
    """Spatio-temporal forecasting network with TCN + GraphSAGE-LSTM.

    Forward contract:
        inputs: (seq_len, batch_size, num_nodes * input_dim)
        node_feas: (num_nodes, node_feat_dim)
        labels: optional, unused for forward but kept for compatibility
        returns (outputs, mid_output, adj_save) where
            outputs: (horizon, batch_size, num_nodes * output_dim)
            mid_output: stacked spatial features (batch, seq_len, num_nodes, hidden_dim)
            adj_save: learned neighbor weights aligned with node_index (num_nodes, num_neighbors)
    """

    def __init__(self, logger, **model_kwargs):
        super().__init__()
        self._logger = logger
        self.num_nodes = int(model_kwargs.get("num_nodes", 1))
        self.input_dim = int(model_kwargs.get("input_dim", 1))
        self.seq_len = int(model_kwargs.get("seq_len", 1))
        self.output_dim = int(model_kwargs.get("output_dim", 1))
        self.horizon = int(model_kwargs.get("horizon", 1))
        self.hidden_dim = int(model_kwargs.get("hidden_dim", model_kwargs.get("rnn_units", 64)))
        self.temporal_backbone = model_kwargs.get("temporal_backbone", "TCN")
        self.spatial_backbone = model_kwargs.get("spatial_backbone", "SAGE_LSTM")
        self.node_feat_dim = int(model_kwargs.get("node_feat_dim", model_kwargs.get("emb_dim", self.input_dim)))
        self.emb_dim = int(model_kwargs.get("emb_dim", self.node_feat_dim))
        self.num_neighbors = int(model_kwargs.get("neighbor_M", model_kwargs.get("num_neighbors", 10)))
        self.multi_scale = bool(model_kwargs.get("multi_scale", False))
        self.use_local = bool(model_kwargs.get("use_local", True))
        self.use_global = bool(model_kwargs.get("use_global", True))
        if not (self.use_local or self.use_global):
            raise ValueError("At least one of use_local or use_global must be True.")
        self.M_local = int(model_kwargs.get("M_local", max(self.num_neighbors, 1)))
        self.M_global = int(model_kwargs.get("M_global", self.num_neighbors))
        self.gate_type = str(model_kwargs.get("gate_type", "scalar")).lower()
        self.temporal_dropout_p = float(model_kwargs.get("temporal_dropout_p", 0.0))
        self.node_dropout_p = float(model_kwargs.get("node_dropout_p", 0.0))
        self.adj_ema_decay = float(model_kwargs.get("adj_ema_decay", 0.99))
        self.adj_ema_mix = float(model_kwargs.get("adj_ema_mix", 1.0))
        self.num_sage_layers = int(model_kwargs.get("num_sage_layers", 2))
        self.tcn_kernel = int(model_kwargs.get("tcn_kernel", 2))
        self.tcn_dropout = float(model_kwargs.get("tcn_dropout", 0.1))
        self.sage_dropout = float(model_kwargs.get("sage_dropout", 0.1))
        tcn_layers = _as_int_list(model_kwargs.get("tcn_layers"), [self.hidden_dim, self.hidden_dim])

        self.input_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.node_mlp = NodeEmbeddingMLP(self.node_feat_dim, self.hidden_dim, self.emb_dim)
        self.node_hidden_proj = nn.Linear(self.emb_dim, self.hidden_dim)
        self.temporal_net = TemporalConvNet(
            num_inputs=self.hidden_dim,
            num_channels=tcn_layers,
            kernel_size=self.tcn_kernel,
            dropout=self.tcn_dropout,
        )
        self.sage = GraphSAGEEncoder(
            in_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_sage_layers,
            num_neighbors=self.num_neighbors,
            dropout=self.sage_dropout,
        )
        self.att_query = nn.Linear(self.emb_dim, self.hidden_dim, bias=False)
        self.att_key = nn.Linear(self.emb_dim, self.hidden_dim, bias=False)
        self.output_head = nn.Linear(self.hidden_dim, self.horizon * self.output_dim)
        gate_out_dim = 1 if self.gate_type == "scalar" else self.hidden_dim
        self.gate_mlp = nn.Linear(self.hidden_dim, gate_out_dim)
        self.res_proj = nn.Identity()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.register_buffer(
            "node_index",
            _init_neighbor_index(self.num_nodes, self.num_neighbors, device),
            persistent=True,
        )
        if self.multi_scale:
            self.register_buffer(
                "node_index_local",
                _init_neighbor_index(self.num_nodes, self.M_local, device),
                persistent=True,
            )
            self.register_buffer(
                "node_index_global",
                _init_neighbor_index(self.num_nodes, self.M_global, device),
                persistent=True,
            )
            self.register_buffer(
                "adj_ema_local",
                torch.zeros(self.num_nodes, self.M_local, device=device),
                persistent=True,
            )
            self.register_buffer(
                "adj_ema_global",
                torch.zeros(self.num_nodes, self.M_global, device=device),
                persistent=True,
            )
        else:
            self.register_buffer("adj_ema_local", torch.zeros(1, 1, device=device), persistent=True)
            self.register_buffer("adj_ema_global", torch.zeros(1, 1, device=device), persistent=True)

    def _reshape_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (seq_len, batch, num_nodes * input_dim)
        seq_len, batch, _ = inputs.shape
        x = inputs.permute(1, 0, 2).contiguous()  # (batch, seq_len, num_nodes * input_dim)
        x = x.reshape(batch, seq_len, self.num_nodes, self.input_dim)
        return x

    def _compute_attention(self, node_embed: torch.Tensor, neighbor_index: torch.Tensor) -> torch.Tensor:
        """Compute slim adjacency weights with attention.

        Args:
            node_embed: (num_nodes, emb_dim)
            neighbor_index: (num_nodes, num_neighbors)
        Returns:
            weights: (num_nodes, num_neighbors) rows sum to 1.
        """
        neighbor_embed = node_embed[neighbor_index]  # (num_nodes, num_neighbors, emb_dim)
        query = self.att_query(node_embed)  # (num_nodes, hidden_dim)
        key = self.att_key(neighbor_embed)  # (num_nodes, num_neighbors, hidden_dim)
        scores = (query.unsqueeze(1) * key).sum(-1) / math.sqrt(self.hidden_dim)
        weights = torch.softmax(scores, dim=1)
        return weights

    def forward(
        self,
        inputs: torch.Tensor,
        node_feas: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        batches_seen: Optional[int] = None,
        batch_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if inputs.dim() != 3:
            raise ValueError(
                f"inputs must be (seq_len, batch, num_nodes*input_dim) but got {inputs.shape}"
            )
        if node_feas.dim() != 2:
            raise ValueError(
                f"node_feas must be (num_nodes, node_feat_dim) but got {node_feas.shape}"
            )
        if node_feas.size(0) != self.num_nodes:
            raise ValueError(
                f"node_feas first dim {node_feas.size(0)} != num_nodes {self.num_nodes}"
            )

        device = inputs.device
        node_feas = node_feas.to(device)
        neighbor_index = self.node_index.to(device)
        neighbor_index_local = None
        neighbor_index_global = None
        if self.multi_scale:
            neighbor_index_local = self.node_index_local.to(device)
            neighbor_index_global = self.node_index_global.to(device)

        x_seq = self._reshape_inputs(inputs)  # (batch, seq_len, num_nodes, input_dim)
        batch_size, seq_len, _, _ = x_seq.shape
        self._contrastive_features = None

        node_embed = self.node_mlp(node_feas)  # (num_nodes, emb_dim)
        neighbor_weights = self._compute_attention(node_embed, neighbor_index)  # (num_nodes, num_neighbors)
        adj_local = None
        adj_global = None
        if self.multi_scale:
            adj_local = self._compute_attention(node_embed, neighbor_index_local)
            adj_global = self._compute_attention(node_embed, neighbor_index_global)
        node_hidden = self.node_hidden_proj(node_embed)  # (num_nodes, hidden_dim)

        if self.training:
            if self.temporal_dropout_p > 0:
                time_mask = torch.rand((batch_size, seq_len, 1, 1), device=device) >= self.temporal_dropout_p
                x_seq = x_seq * time_mask
            if self.node_dropout_p > 0:
                node_mask = torch.rand((batch_size, 1, self.num_nodes, 1), device=device) >= self.node_dropout_p
                x_seq = x_seq * node_mask
            if self.multi_scale:
                with torch.no_grad():
                    self.adj_ema_local.mul_(self.adj_ema_decay).add_((1 - self.adj_ema_decay) * adj_local.detach())
                    self.adj_ema_global.mul_(self.adj_ema_decay).add_((1 - self.adj_ema_decay) * adj_global.detach())
                adj_local = self.adj_ema_mix * adj_local + (1 - self.adj_ema_mix) * self.adj_ema_local
                adj_global = self.adj_ema_mix * adj_global + (1 - self.adj_ema_mix) * self.adj_ema_global

        # Vectorized spatial encoding over all time steps to avoid Python loops.
        x_proj = self.input_proj(x_seq)  # (batch, seq_len, num_nodes, hidden_dim)
        node_embed_expanded = node_hidden.view(1, 1, self.num_nodes, self.hidden_dim)
        fused_input = x_proj + node_embed_expanded  # broadcast static embeddings
        fused_flat = fused_input.reshape(batch_size * seq_len, self.num_nodes, self.hidden_dim)

        if self.multi_scale:
            base_res = self.res_proj(fused_flat)
            h_local = None
            h_global = None
            if self.use_local:
                h_local = self.sage(fused_flat, neighbor_index_local, adj_local)
            if self.use_global:
                h_global = self.sage(fused_flat, neighbor_index_global, adj_global)
            if h_local is None:
                h_flat = base_res + h_global
            elif h_global is None:
                h_flat = base_res + h_local
            else:
                gate = torch.sigmoid(self.gate_mlp(fused_flat))
                if gate.shape[-1] == 1:
                    gate = gate.expand_as(h_local)
                h_fused = gate * h_local + (1 - gate) * h_global
                h_flat = base_res + h_fused
        else:
            h_flat = self.sage(fused_flat, neighbor_index, neighbor_weights)  # (batch*seq, num_nodes, hidden_dim)

        spatial_stack = h_flat.reshape(batch_size, seq_len, self.num_nodes, self.hidden_dim)

        # TCN expects (batch*num_nodes, hidden_dim, seq_len)
        tcn_in = spatial_stack.permute(0, 2, 3, 1).contiguous()
        tcn_in = tcn_in.reshape(batch_size * self.num_nodes, self.hidden_dim, seq_len)
        tcn_out = self.temporal_net(tcn_in)
        tcn_out = tcn_out.reshape(batch_size, self.num_nodes, self.hidden_dim, seq_len)
        tcn_out = tcn_out.permute(0, 3, 1, 2).contiguous()  # (batch, seq_len, num_nodes, hidden_dim)
        self._contrastive_features = tcn_out.mean(dim=1)  # (batch, num_nodes, hidden_dim)

        last_hidden = tcn_out[:, -1, :, :]  # (batch, num_nodes, hidden_dim)
        logits = self.output_head(last_hidden)  # (batch, num_nodes, horizon*output_dim)
        logits = logits.reshape(batch_size, self.num_nodes, self.horizon, self.output_dim)
        logits = logits.permute(2, 0, 1, 3).contiguous()
        outputs = logits.reshape(self.horizon, batch_size, self.num_nodes * self.output_dim)

        if self.multi_scale:
            adj_save = {"local": adj_local, "global": adj_global}
        else:
            adj_save = neighbor_weights
        mid_output = spatial_stack
        if batches_seen == 0:
            total_params = sum(p.numel() for p in self.parameters())
            self._logger.info(
                "SAGDFNModel params: hidden_dim=%d, neighbors=%d, total_params=%d"
                % (self.hidden_dim, self.num_neighbors, total_params)
            )

        return outputs, mid_output, adj_save

    def get_contrastive_features(self) -> Optional[torch.Tensor]:
        """Return pooled TCN features for optional contrastive losses. Shape: (batch, num_nodes, hidden_dim)."""
        return self._contrastive_features