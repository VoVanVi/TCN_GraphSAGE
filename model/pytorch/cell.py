import numpy as np
import torch
import torch.nn as nn
from lib import utils
import time
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class LayerParams:
    def __init__(self, rnn_network: torch.nn.Module, layer_type: str):
        self._rnn_network = rnn_network
        self._params_dict = {}
        self._biases_dict = {}
        self._type = layer_type

    def get_weights(self, shape):
        if shape not in self._params_dict:
            nn_param = torch.nn.Parameter(torch.empty(*shape, device=device))
            torch.nn.init.xavier_normal_(nn_param)
            self._params_dict[shape] = nn_param
            self._rnn_network.register_parameter('{}_weight_{}'.format(self._type, str(shape)),
                                                 nn_param)
        return self._params_dict[shape]

    def get_biases(self, length, bias_start=0.0):
        if length not in self._biases_dict:
            biases = torch.nn.Parameter(torch.empty(length, device=device))
            torch.nn.init.constant_(biases, bias_start)
            self._biases_dict[length] = biases
            self._rnn_network.register_parameter('{}_biases_{}'.format(self._type, str(length)),
                                                 biases)

        return self._biases_dict[length]


class DCGRUCell(torch.nn.Module): # OneStepFastGConv cell use DCRNN DCGRUCell as backbone
    def __init__(self, num_units, max_diffusion_step, num_nodes, nonlinearity='tanh',
                 filter_type="laplacian", use_gc_for_ru=True,
                 aggregation_type='diffusion', graphsage_neighbors=None):
        """

        :param num_units:
        :param adj_mx:
        :param max_diffusion_step:
        :param num_nodes:
        :param nonlinearity:
        :param filter_type: "laplacian", "random_walk", "dual_random_walk".
        :param use_gc_for_ru: whether to use Graph convolution to calculate the reset and update gates.
        """

        super().__init__()
        self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu
        # support other nonlinearities up here?
        self._num_nodes = num_nodes
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._supports = []
        self._use_gc_for_ru = use_gc_for_ru
        self._aggregation_type = aggregation_type
        self._graphsage_neighbors = graphsage_neighbors
        self._cached_graphsage_neighbors = None
        self._cached_graphsage_source = None
        
        '''
        Option:
        if filter_type == "laplacian":
            supports.append(utils.calculate_scaled_laplacian(adj_mx, lambda_max=None))
        elif filter_type == "random_walk":
            supports.append(utils.calculate_random_walk_matrix(adj_mx).T)
        elif filter_type == "dual_random_walk":
            supports.append(utils.calculate_random_walk_matrix(adj_mx).T)
            supports.append(utils.calculate_random_walk_matrix(adj_mx.T).T)
        else:
            supports.append(utils.calculate_scaled_laplacian(adj_mx))
        for support in supports:
            self._supports.append(self._build_sparse_matrix(support))
        '''

        self._fc_params = LayerParams(self, 'fc')
        self._gconv_params = LayerParams(self, 'gconv')
        self._graphsage_aggregators = {}

    @staticmethod
    def _build_sparse_matrix(L):
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        # this is to ensure row-major ordering to equal torch.sparse.sparse_reorder(L)
        indices = indices[np.lexsort((indices[:, 0], indices[:, 1]))]
        L = torch.sparse_coo_tensor(indices.T, L.data, L.shape, device=device)
        return L

    def _calculate_random_walk_matrix(self, adj_mx):

        d = torch.sum(adj_mx, 1)+1
        d_inv = 1. / d
        d_inv = torch.where(torch.isinf(d_inv), torch.zeros(d_inv.shape).to(device), d_inv)
        d_mat_inv = torch.diag(d_inv)
        random_walk_mx = torch.mm(d_mat_inv, adj_mx)
        return random_walk_mx, d_mat_inv
    
    def _calculate_random_walk_matrix_full(self, adj_mx):
        adj_mx = adj_mx + torch.eye(int(adj_mx.shape[0])).to(device)
        d = torch.sum(adj_mx, 1)
        d_inv = 1. / d
        d_inv = torch.where(torch.isinf(d_inv), torch.zeros(d_inv.shape).to(device), d_inv)
        d_mat_inv = torch.diag(d_inv)
        random_walk_mx = torch.mm(d_mat_inv, adj_mx)
        return random_walk_mx

    def _graphsage_neighbors_from_dense(self, adj_mx):
        if adj_mx.dim() != 2 or adj_mx.shape[0] != adj_mx.shape[1]:
            self._cached_graphsage_neighbors = None
            self._cached_graphsage_source = None
            return None

        cache_key = (adj_mx.data_ptr(), adj_mx._version, adj_mx.shape)
        if self._cached_graphsage_source == cache_key and self._cached_graphsage_neighbors is not None:
            return self._cached_graphsage_neighbors

        dense_adj = adj_mx.clone()
        if dense_adj.shape[0] == self._num_nodes:
            dense_adj = dense_adj.fill_diagonal_(0.0)

        if self._graphsage_neighbors is None or self._graphsage_neighbors >= dense_adj.shape[1]:
            neighbor_weights = dense_adj
            neighbor_index = torch.arange(self._num_nodes, device=device).unsqueeze(0).repeat(self._num_nodes, 1)
        else:
            topk = torch.topk(dense_adj, self._graphsage_neighbors, dim=1)
            neighbor_weights = topk.values
            neighbor_index = topk.indices

        self._cached_graphsage_source = cache_key
        self._cached_graphsage_neighbors = (neighbor_index, neighbor_weights)
        return neighbor_index, neighbor_weights
    
    def _message_passing(self, adj_mx, d_mat_inv, inputs, node_index):
        

        output = torch.mm(adj_mx, inputs[node_index,:])+torch.mm(d_mat_inv, inputs)

        return output

    def _get_graphsage_lstm(self, input_size):
        if input_size not in self._graphsage_aggregators:
            lstm = nn.LSTM(input_size, input_size, batch_first=True).to(device)
            self._graphsage_aggregators[input_size] = lstm
            self.add_module(f'graphsage_lstm_{input_size}', lstm)
        return self._graphsage_aggregators[input_size]

    def _graphsage_aggregate(self, features, adj_mx, node_index, input_size):
        batch_size, num_nodes, feat_dim = features.shape
        sampled = self._graphsage_neighbors_from_dense(adj_mx)
        if sampled is None:
            neighbor_index = node_index
            neighbor_weights = adj_mx
        else:
            neighbor_index, neighbor_weights = sampled

        neighbor_features = features[:, neighbor_index, :]
        neighbor_features = neighbor_features * neighbor_weights.unsqueeze(0).unsqueeze(-1)
        lstm_input = neighbor_features.view(batch_size * num_nodes, neighbor_features.size(2), feat_dim)
        aggregator = self._get_graphsage_lstm(input_size)
        _, (hidden_state, _) = aggregator(lstm_input)
        aggregated = hidden_state[-1].view(batch_size, num_nodes, feat_dim)
        return aggregated

    def forward(self, inputs, hx, adj, node_index):
        """Gated recurrent unit (GRU) with Graph Convolution.
        :param inputs: (B, num_nodes * input_dim)
        :param hx: (B, num_nodes * rnn_units)

        :return
        - Output: A `2-D` tensor with shape `(B, num_nodes * rnn_units)`.
        """

        if adj.shape[0] == adj.shape[1]:
            adj_mx = self._calculate_random_walk_matrix_full(adj).t()
            d_mat_inv = torch.eye(adj_mx.shape[0])
        else:
            adj_mx, d_mat_inv = self._calculate_random_walk_matrix(adj)
        output_size = 2 * self._num_units
        if self._use_gc_for_ru:
            fn = self._gconv
        else:
            fn = self._fc
        value = torch.sigmoid(fn(inputs, adj_mx, node_index,d_mat_inv, hx, output_size, bias_start=1.0))
        value = torch.reshape(value, (-1, self._num_nodes, output_size))
        r, u = torch.split(tensor=value, split_size_or_sections=self._num_units, dim=-1)
        r = torch.reshape(r, (-1, self._num_nodes * self._num_units))
        u = torch.reshape(u, (-1, self._num_nodes * self._num_units))

        c = self._gconv(inputs, adj_mx, node_index,d_mat_inv, r * hx, self._num_units)
        if self._activation is not None:
            c = self._activation(c)

        new_state = u * hx + (1.0 - u) * c
        return new_state

    @staticmethod
    def _concat(x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)

    def _fc(self, inputs, state, output_size, bias_start=0.0):
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size * self._num_nodes, -1))
        state = torch.reshape(state, (batch_size * self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=-1)
        input_size = inputs_and_state.shape[-1]
        weights = self._fc_params.get_weights((input_size, output_size))
        value = torch.sigmoid(torch.matmul(inputs_and_state, weights))
        biases = self._fc_params.get_biases(output_size, bias_start)
        value += biases
        return value

    def _gconv(self, inputs, adj_mx, node_index,d_mat_inv,  state, output_size, bias_start=0.0):
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = torch.reshape(state, (batch_size, self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = inputs_and_state.size(2)

        if self._aggregation_type == 'graphsage':
            aggregated = self._graphsage_aggregate(inputs_and_state, adj_mx, node_index, input_size)
            x = torch.cat([inputs_and_state, aggregated], dim=2)
            x = torch.reshape(x, shape=[batch_size * self._num_nodes, x.size(2)])
            weights = self._gconv_params.get_weights((x.size(1), output_size))
        else:
            x = inputs_and_state
            x0 = x.permute(1, 2, 0)  # (num_nodes, total_arg_size, batch_size)
            x0 = torch.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
            x = torch.unsqueeze(x0, 0)
            if self._max_diffusion_step != 0:
                if adj_mx.shape[0] == adj_mx.shape[1]:
                    x1 = torch.mm(adj_mx, x0)
                else:
                    x1 = self._message_passing(adj_mx,d_mat_inv, x0, node_index)
                x = self._concat(x, x1)

                for k in range(2, self._max_diffusion_step + 1):
                    if adj_mx.shape[0] == adj_mx.shape[1]:
                        x2 = 2 * torch.mm(adj_mx, x1) - x0
                    else:
                        x2 = 2 * self._message_passing(adj_mx, d_mat_inv, x1, node_index) - x0
                    x = self._concat(x, x2)
                    x1, x0 = x2, x1
            num_matrices = self._max_diffusion_step + 1  # Adds for x itself.
            x = torch.reshape(x, shape=[num_matrices, self._num_nodes, input_size, batch_size])
            x = x.permute(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, order)
            x = torch.reshape(x, shape=[batch_size * self._num_nodes, input_size * num_matrices])
            weights = self._gconv_params.get_weights((input_size * num_matrices, output_size))

        x = torch.matmul(x, weights)  # (batch_size * self._num_nodes, output_size)
        biases = self._gconv_params.get_biases(output_size, bias_start)
        x += biases
        return torch.reshape(x, [batch_size, self._num_nodes * output_size])
