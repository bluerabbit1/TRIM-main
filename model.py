import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.utils import softmax, degree
from torch_geometric.nn import GCNConv
from torch_scatter import scatter
import math


class RelTemporalEncoding(nn.Module):
    def __init__(self, n_hid, max_len=50, dropout=0.2):
        super(RelTemporalEncoding, self).__init__()
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_hid, 2) *
                             -(math.log(10000.0) / n_hid))
        emb = nn.Embedding(max_len, n_hid)
        emb.weight.data[:, 0::2] = torch.sin(position * div_term) / math.sqrt(n_hid)
        emb.weight.data[:, 1::2] = torch.cos(position * div_term) / math.sqrt(n_hid)
        emb.requires_grad = False
        self.emb = emb
        self.lin = nn.Linear(n_hid, n_hid)

    def forward(self, x, t):
        if isinstance(t, int):
            t = torch.arange(t).to(x.device)
            enc = self.lin(self.emb(t))
            enc = enc.expand_as(x)
            return x + enc
        else:
            return x + self.lin(self.emb(t))


class StructuralAttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, n_heads, dropout, residual,
                 use_fmask=False, norm=True, skip=False):
        super(StructuralAttentionLayer, self).__init__()
        self.out_dim = output_dim // n_heads
        self.n_heads = n_heads
        self.in_dim, self.hid_dim = input_dim, output_dim
        self.update_norm = nn.LayerNorm(output_dim)
        self.cs_mlp = nn.Sequential(
            nn.Linear(output_dim, 2 * output_dim), nn.GELU(),
            nn.Linear(2 * output_dim, output_dim))
        self.fmask = nn.Parameter(torch.ones(output_dim))
        self.residual = residual
        if self.residual:
            self.lin_residual = nn.Linear(input_dim, n_heads * self.out_dim, bias=False)
        self.q_linear = nn.Linear(input_dim, output_dim)
        self.k_linear = nn.Linear(input_dim, output_dim)
        self.v_linear = nn.Linear(input_dim, output_dim)
        self.d_k = output_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.aggr = 'add'
        self.update_drop = nn.Dropout(dropout)
        self.update_skip = nn.Parameter(torch.ones(1))
        self.update_linear = nn.Linear(output_dim, output_dim)
        self.use_fmask = use_fmask
        self.norm = norm
        self.skip = skip
        self.node_dim = 0

    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.shape[1]).view(-1, 1).to(x.device)
        q_mat = self.q_linear(x[edge_index[1]]).view(-1, self.n_heads, self.d_k)
        k_mat = self.k_linear(x[edge_index[0]]).view(-1, self.n_heads, self.d_k)
        v_mat = self.v_linear(x[edge_index[0]]).view(-1, self.n_heads, self.d_k)
        res_att = (q_mat * k_mat).sum(dim=-1) / self.sqrt_dk
        res_att = edge_weight.view(-1, 1) * res_att
        res_msg = v_mat
        ei_tar = edge_index[1]
        res_att = softmax(res_att, ei_tar)
        res = res_msg * res_att.view(-1, self.n_heads, 1)
        res = res.view(-1, self.hid_dim)
        res = scatter(res, ei_tar, dim=self.node_dim,
                      dim_size=x.shape[0], reduce=self.aggr)
        if self.use_fmask:
            fmask_c = F.softmax(self.fmask, dim=0)
            res = res * fmask_c

        def ffn(x):
            if self.norm:
                res = self.cs_mlp(self.update_norm(x))
            else:
                res = self.cs_mlp(x)
            res = self.update_drop(res)
            if self.skip:
                alpha = torch.sigmoid(self.update_skip)
                res = (1 - alpha) * x + alpha * res
            else:
                res = x + res
            return res

        res = ffn(res + x)
        return res


class TemporalAttentionLayer(nn.Module):
    def __init__(self, input_dim, n_heads, attn_drop, residual, use_RTE=True):
        super(TemporalAttentionLayer, self).__init__()
        self.n_heads = n_heads
        self.residual = residual
        self.RTE = use_RTE
        self.time_emb = RelTemporalEncoding(input_dim)
        self.Q_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.K_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.V_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.lin = nn.Linear(input_dim, input_dim, bias=True)
        self.attn_dp = nn.Dropout(attn_drop)
        self.xavier_init()

    def forward(self, inputs, time_encoding):
        time_length = inputs.shape[1]
        if self.RTE:
            temporal_inputs = time_encoding(inputs, time_length)
        else:
            temporal_inputs = inputs
        q = torch.tensordot(temporal_inputs, self.Q_embedding_weights, dims=([2], [0]))
        k = torch.tensordot(temporal_inputs, self.K_embedding_weights, dims=([2], [0]))
        v = torch.tensordot(temporal_inputs, self.V_embedding_weights, dims=([2], [0]))
        split_size = int(q.shape[-1] / self.n_heads)
        q_ = torch.cat(torch.split(q, split_size_or_sections=split_size, dim=2), dim=0)
        k_ = torch.cat(torch.split(k, split_size_or_sections=split_size, dim=2), dim=0)
        v_ = torch.cat(torch.split(v, split_size_or_sections=split_size, dim=2), dim=0)
        outputs = torch.matmul(q_, k_.permute(0, 2, 1))
       # outputs = outputs / (time_length ** 0.5)
        outputs = outputs / (split_size ** 0.5)
        diag_val = torch.ones_like(outputs[0])
        tril = torch.tril(diag_val)
        masks = tril[None, :, :].repeat(outputs.shape[0], 1, 1)
        padding = torch.ones_like(masks) * (-2 ** 32 + 1)
        outputs = torch.where(masks == 0, padding, outputs)
        outputs = F.softmax(outputs, dim=2)
        self.attn_wts_all = outputs
        if self.training:
            outputs = self.attn_dp(outputs)
        outputs = torch.matmul(outputs, v_)
        outputs = torch.cat(
            torch.split(outputs,
                        split_size_or_sections=int(outputs.shape[0] / self.n_heads),
                        dim=0),
            dim=2)
        outputs = self.feedforward(outputs)
        if self.residual:
            outputs = outputs + temporal_inputs
        return outputs

    def feedforward(self, inputs):
        outputs = F.relu(self.lin(inputs))
        return outputs + inputs

    def xavier_init(self):
        nn.init.xavier_uniform_(self.Q_embedding_weights)
        nn.init.xavier_uniform_(self.K_embedding_weights)
        nn.init.xavier_uniform_(self.V_embedding_weights)


class TRIM(nn.Module):
    def __init__(self, args):
        super(TRIM, self).__init__()
        self.args = args

        self.feat = Parameter(
            (torch.ones(args.num_nodes, args.input_dim)).to(args.device),
            requires_grad=True)
        self.device = args.device
        self.structural_head_config = list(map(int, args.structural_head_config.split(",")))
        self.structural_layer_config = list(map(int, args.structural_layer_config.split(",")))
        self.temporal_head_config = list(map(int, args.temporal_head_config.split(",")))
        self.temporal_layer_config = list(map(int, args.temporal_layer_config.split(",")))
        self.spatial_drop = args.spatial_drop
        self.temporal_drop = args.temporal_drop

        self.linear = nn.Linear(args.input_dim, args.hid_dim, bias=bool(args.lin_bias))
        self.time_emb = RelTemporalEncoding(args.hid_dim)
        self.structural_attn, self.temporal_attn = self.build_model()

        if args.experiment_name == 'lp':
            self.cs_decoder = MultiplyPredictor()
        else:
            self.cs_decoder = NodeClf(args.nc_layers, args.num_classes, args.hid_dim)

    def forward(self, edge_index_list, x_list, train_len):
        if x_list is None:
            x = [self.linear(self.feat) for i in range(len(edge_index_list))]
        else:
            x = [self.linear(xi) for xi in x_list]

        structural_out = []
        for t in range(train_len):
            edge_index_t = edge_index_list[t].to(x[t].device)

            # Spatial attention (uniform edge weight, no EdgeScorer prior)
            for j, layer in enumerate(self.structural_attn):
                if j == 0:
                    spatial_out = layer(x[t], edge_index_t)
                else:
                    spatial_out = layer(spatial_out, edge_index_t)
                if j != len(self.structural_attn) - 1:
                    spatial_out = F.relu(spatial_out)
            structural_out.append(spatial_out)

        # Temporal attention
        structural_outputs = torch.stack([g for g in structural_out], dim=1)  # [N, T, F]

        for j, layer in enumerate(self.temporal_attn):
            if j == 0:
                temporal_out = layer(structural_outputs, self.time_emb)
            else:
                temporal_out = layer(temporal_out, self.time_emb)

        return temporal_out

    def build_model(self):
        input_dim = self.args.hid_dim
        structural_attention_layers = nn.ModuleList()
        for i in range(len(self.structural_layer_config)):
            layer = StructuralAttentionLayer(
                input_dim=input_dim,
                output_dim=self.structural_layer_config[i],
                n_heads=self.structural_head_config[i],
                dropout=self.args.spatial_drop,
                residual=self.args.residual,
                use_fmask=self.args.fmask,
                norm=self.args.norm,
                skip=self.args.skip)
            structural_attention_layers.add_module(
                name="structural_layer_{}".format(i), module=layer)
            input_dim = self.structural_layer_config[i]

        input_dim = self.structural_layer_config[-1]
        temporal_attention_layers = nn.ModuleList()
        for i in range(len(self.temporal_layer_config)):
            layer = TemporalAttentionLayer(
                input_dim=input_dim,
                n_heads=self.temporal_head_config[i],
                attn_drop=self.temporal_drop,
                residual=self.args.residual,
                use_RTE=self.args.use_RTE)
            temporal_attention_layers.add_module(
                name="temporal_layer_{}".format(i), module=layer)
            input_dim = self.temporal_layer_config[i]
        return structural_attention_layers, temporal_attention_layers


# =============================================================================
# Decoders
# =============================================================================

class MultiplyPredictor(torch.nn.Module):
    def __init__(self, temperature=1.0):
        super(MultiplyPredictor, self).__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self, z, edge_index):
        z = F.normalize(z, p=2, dim=1)
        pred = torch.sum(z[edge_index[0]] * z[edge_index[1]], dim=1)
        return torch.sigmoid(pred / self.temperature)


class NodeClf(nn.Module):
    def __init__(self, layers, num_class, hid_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(hid_dim, num_class))

    def forward(self, x):
        for layer in self.clf:
            x = layer(x)
        return x

