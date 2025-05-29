import numpy as np
import dgl
from dgl.nn.pytorch import TypedLinear, GraphConv
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.ops import MLPReadout

class FFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FFN, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.w_2(self.dropout(self.relu(self.w_1(x))))

class embedding(nn.Module):
    def __init__(self, in_dim):
        super(embedding, self).__init__()
        k = 3
        self.in_dim = in_dim
        ffn_ratio = 2
        self.Avgpool1 = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.AvgPool1d(k, stride=k),
            torch.nn.Dropout(0.1)
        )
        self.ConvFFN = torch.nn.Sequential(
            torch.nn.BatchNorm1d(self.in_dim),
            torch.nn.Conv1d(self.in_dim, self.in_dim * ffn_ratio, kernel_size=1, stride=1, padding=0, groups=1),
            torch.nn.GELU(),
            torch.nn.Conv1d(self.in_dim * ffn_ratio, self.in_dim, kernel_size=1, stride=1, padding=0, groups=1),
        )
        self.Avgpool2 = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.AvgPool1d(k, stride=k),
            torch.nn.Dropout(0.1)
        )

    def forward(self, outputs):
        outputs = outputs.transpose(1, 2)
        outputs = self.Avgpool1(outputs)
        outputs += self.ConvFFN(outputs)
        outputs = self.Avgpool2(outputs)
        '''
              Layer2
        '''
        outputs = outputs.transpose(1, 2)
        outputs = outputs.sum(dim=1)
        return outputs


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, num_heads=4, num_layers=1, hidden_size=100, d_ff=512, dropout=0.1):
        super(MultiHeadCrossAttention, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=num_heads, batch_first=True)
            for _ in range(self.num_layers)
        ])
        self.self_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=num_heads, batch_first=True)
            for _ in range(self.num_layers)
        ])
        self.ffn_layers = nn.ModuleList([
            FFN(d_model=self.hidden_size, d_ff=d_ff, dropout=dropout)
            for _ in range(self.num_layers)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(self.hidden_size) for _ in range(num_layers * 2)])
        self.dropout = nn.Dropout(dropout)
        # 引入可学习的权重参数 alpha
        self.alpha = nn.Parameter(torch.ones(num_layers))

    def forward(self, x, y):
        for i in range(self.num_layers):
            cross_attn = self.cross_attn_layers[i]
            self_attn = self.self_attn_layers[i]

            ffn = self.ffn_layers[i]
            norm1 = self.layer_norms[i * 2]
            norm2 = self.layer_norms[i * 2 + 1]

            cross_attn_output, _ = cross_attn(x, y, y)
            self_attn_output, _ = self_attn(x, x, x)

            alpha_i = torch.sigmoid(self.alpha[i])
            attn_output = alpha_i * cross_attn_output + (1 - alpha_i) * self_attn_output
            attn_output = self.dropout(attn_output)  # 应用 Dropout
            x = norm1(x + attn_output)

            ffn_output = ffn(x)

            x = norm2(x + ffn_output)

        return x

class Pool(nn.Module):

    def __init__(self, k, max_edge_types,l_dim):
        super(Pool, self).__init__()
        self.k = k
        self.dropout = nn.Dropout(0.1)
        self.max_edge_types = max_edge_types
        self.l_dim = l_dim
        self.typed_linear = TypedLinear(self.l_dim * 2, self.l_dim, self.max_edge_types, regularizer='basis', num_bases=2)
        self.w_q = nn.Linear(self.l_dim, self.l_dim)
        self.w_k = nn.Linear(self.l_dim, self.l_dim)
        self.w_v = nn.Linear(self.l_dim, 1)
    def forward(self, g):
        g = self.compute_edge_attention(g)
        g.ndata['orig_nid'] = torch.arange(g.num_nodes()).to(g.device)
        g_list = dgl.unbatch(g)
        original_node_ids_list = []
        updated_h_list = []
        # 存储每个子图的原始节点索引
        for subg in g_list:
            node_attention = subg.ndata['y']
            if torch.isnan(node_attention).any() or torch.isinf(node_attention).any():
                raise ValueError("node_attention contains NaN or Inf values")
            # 动态计算保留的节点数
            num_nodes_to_keep = int(self.k * subg.num_nodes())
            if num_nodes_to_keep < 2:  # 至少保留2个节点
                num_nodes_to_keep = 2
            topk_indices = torch.topk(node_attention, num_nodes_to_keep, dim=0, sorted=False).indices
            topk_indices = topk_indices.view(-1)
            sorted_values, sorted_indices = torch.sort(topk_indices, descending=False)
            # 获取子图中节点在原始图中的索引
            original_node_ids = subg.ndata['orig_nid'][sorted_values]
            original_node_ids_list.append(original_node_ids)

            selected_y = node_attention[sorted_indices]
            selected_h = subg.ndata['h'][sorted_indices]
            attention_weights = torch.sigmoid(selected_y)
            # 更新 h 属性
            updated_h = selected_h * attention_weights
            updated_h = self.dropout(updated_h)  # 应用 Dropout
            updated_h_list.append(updated_h)

        # 将所有子图的原始节点索引合并为一个张量
        all_original_node_ids = torch.cat(original_node_ids_list, dim=0)
        final_subg = dgl.node_subgraph(g, all_original_node_ids)
        # 将更新后的 h 属性合并到最终子图中
        final_subg.ndata['h'] = torch.cat(updated_h_list, dim=0)
        return final_subg
    def edge_embedding(self):
        def func(edges):
            edge_type = edges.data['etype']
            z2 = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
            edge_embedding = self.typed_linear(z2, edge_type)
            n_k = self.w_k(edge_embedding)
            n_v = self.w_v(edge_embedding)
            return {'n_k': n_k, 'n_v': n_v}

        return func

    def e_dot_dst(self, src_field, dst_field, out_field):
        def func(edges):
            return {out_field: (edges.data[src_field] * edges.dst[dst_field]).sum(-1, keepdim=True)}

        return func

    def scaled_exp(self, field, scale):
        def func(edges):
            return {field: torch.exp((edges.data[field] / scale).clamp(-10, 10))}

        return func

    def calculate_edge_value(self, score, value, out_field):
        def func(edges):
            return {out_field: edges.data[score] * edges.data[value]}

        return func

    def compute_edge_attention(self, g):
        # 计算自环
        h = g.ndata['h']
        # 自环的嵌入表示，通过将节点特征与其自身拼接来实现
        self_embedding = torch.cat([h, h], dim=1)
        # 假设所有自环的类型是 4
        etype = torch.full((g.num_nodes(),), fill_value=4, dtype=torch.long, device=h.device)
        # 对自环的嵌入表示进行线性变换
        transformed_embedding = self.typed_linear(self_embedding, etype)
        # 通过 w_v 线性层生成最终的节点表示
        self_y = self.w_v(transformed_embedding)

        # 正向图：计算入边邻居的重要性分数
        g_clone = g.clone()
        g_clone = dgl.remove_self_loop(g_clone)
        n_q = self.w_q(g_clone.ndata['h'])
        g_clone.ndata['n_q'] = n_q
        g_clone.apply_edges(self.edge_embedding())
        g_clone.apply_edges(self.e_dot_dst('n_k', 'n_q', 'in_score'))  # 入边分数
        g_clone.apply_edges(self.scaled_exp('in_score', np.sqrt(self.l_dim)))
        g_clone.apply_edges(self.calculate_edge_value('in_score', 'n_v', 'in_e'))
        g_clone.update_all(fn.copy_e('in_e', 'wV'), fn.sum('wV', 'in_y'))
        g_clone.update_all(fn.copy_e('in_score', 'in_score'), fn.sum('in_score', 'in_z'))
        g_clone.ndata['in_y'] = g_clone.ndata['in_y'] / (
                    g_clone.ndata['in_z'] + torch.full_like(g_clone.ndata['in_z'], 1e-6))
        in_y = g_clone.ndata.pop('in_y')

        # 反向图：计算出边邻居的重要性分数
        g_rev = dgl.reverse(g, copy_edata=True)
        g_rev = dgl.remove_self_loop(g_rev)
        rev_q = self.w_q(g_rev.ndata['h'])
        g_rev.ndata['rev_q'] = rev_q
        g_rev.apply_edges(self.edge_embedding())
        g_rev.apply_edges(self.e_dot_dst('n_k', 'rev_q', 'out_score'))  # 出边分数
        g_rev.apply_edges(self.scaled_exp('out_score', np.sqrt(self.l_dim)))
        g_rev.apply_edges(self.calculate_edge_value('out_score', 'n_v', 'out_e'))
        g_rev.update_all(fn.copy_e('out_e', 'wV'), fn.sum('wV', 'out_y'))
        g_rev.update_all(fn.copy_e('out_score', 'out_score'), fn.sum('out_score', 'out_z'))
        g_rev.ndata['out_y'] = g_rev.ndata['out_y'] / (
                    g_rev.ndata['out_z'] + torch.full_like(g_rev.ndata['out_z'], 1e-6))
        out_y = g_rev.ndata.pop('out_y')

        # 综合入边和出边的结果
        g.ndata['y'] = in_y + out_y + self_y
        return g


class   EA_Unet(nn.Module):
    def __init__(self, ks, l_num, l_dim, dropout, max_edge_types):
        super(EA_Unet, self).__init__()
        self.ks = ks
        self.l_dim = l_dim
        self.dropout = dropout
        self.max_edge_types = max_edge_types
        self.l_num = l_num
        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.gcn = GraphConv(l_dim, l_dim, activation=F.relu)
        self.dropout_layer = nn.Dropout(dropout)  # 添加 Dropout
        for i in range(self.l_num):
            self.down_layers.append(GraphConv(l_dim, l_dim, activation=F.relu))
            self.up_layers.append(GraphConv(l_dim, l_dim, activation=F.relu))
            self.pools.append(Pool(ks,  max_edge_types, l_dim))

    def forward(self, g, h):
        multi_scale_graphs = []
        for i in range(self.l_num):
            h = self.down_layers[i](g, h)
            h = self.dropout_layer(h)  # 应用 Dropout
            g.ndata['h'] = h
            multi_scale_graphs.append(g.clone())
            g = self.pools[i](g)
            h = g.ndata['h']
        h = self.gcn(g, h)
        h = self.dropout_layer(h)  # 应用 Dropout
        g.ndata['h'] = h
        multi_scale_graphs.append(g)

        for i in range(len(multi_scale_graphs) - 1, 0, -1):
            father = multi_scale_graphs[i - 1]
            child = multi_scale_graphs[i]
            f_h = father.ndata['h']
            c_h = child.ndata['h']
            combined_h = torch.zeros_like(f_h)
            original_node_ids = child.ndata[dgl.NID]
            combined_h[original_node_ids] = c_h
            combined_h = f_h + combined_h
            enhenched_h = self.up_layers[i - 1](father, combined_h)
            father.ndata['h'] = enhenched_h
        return multi_scale_graphs[0]

class GLHFE_CSGVD(nn.Module):
    def __init__(self, text_encoder, in_dim, args):
        super(GLHFE_CSGVD, self).__init__()
        self.text_encoder = text_encoder
        self.args = args
        self.in_dim = in_dim
        self.layernorm_t = nn.LayerNorm(normalized_shape=args.l_dim)
        self.layernorm_g = nn.LayerNorm(normalized_shape=args.l_dim)
        self.cross_attention_text_to_graph = MultiHeadCrossAttention(hidden_size=args.l_dim)
        self.cross_attention_graph_to_text = MultiHeadCrossAttention(hidden_size=args.l_dim)
        self.proj_g = nn.Linear(in_dim, args.l_dim)
        self.proj_t = nn.Linear(768, args.l_dim)
        self.g_unet = EA_Unet(
            args.ks, args.l_num,args.l_dim, args.drop_n, args.max_edge_types)
        self.embed_t = embedding(args.l_dim)
        self.embed_g = embedding(args.l_dim)
        self.act = nn.ReLU()
        self.mlp = MLPReadout(args.l_dim * 2, 2)

    def to_cuda(self, x):
        if isinstance(x, torch.Tensor):
            return x.cuda()
        elif isinstance(x, list):
            return [self.to_cuda(item) for item in x]
        else:
            return x

    def forward(self, gs, hs, texts, ys):
        texts = torch.tensor(texts)
        gs, hs, texts, ys = map(self.to_cuda, [gs, hs, texts, ys])
        # text encoding
        trans_texts = self.text_encoder(texts)
        texts = self.proj_t(trans_texts)
        texts = self.act(texts)
        texts = self.layernorm_t(texts)

        # graph encoding
        for i, (g, h) in enumerate(zip(gs, hs)):
            gs[i] = g.to(ys.device)  # 更新 gs 中的图对象
            h = self.proj_g(h)
            h = self.act(h)
            h = self.layernorm_g(h)
            gs[i].ndata['h'] = h
        batch_g = dgl.batch(gs)
        graphs = self.embed(batch_g)

        # cross attention
        g_embed = self.cross_attention_text_to_graph(graphs, texts)
        t_embed = self.cross_attention_graph_to_text(texts, graphs)
        logits = self.classify(g_embed, t_embed)
        return self.metric(logits, ys)

    def classify(self, g, t):
        t = self.embed_t(t)
        g = self.embed_g(g)
        outputs = torch.cat([g, t], dim=1)
        outputs = self.mlp(outputs)
        logits = F.softmax(outputs, dim=1)
        return logits

    def metric(self, logits, labels):
        loss = F.cross_entropy(logits, labels.long())
        _, preds = torch.max(logits, 1)
        return loss, preds

    def embed(self, g):
        h = g.ndata['h']
        g = self.g_unet(g, h)
        hs = self.readout(g)
        return hs

    def readout(self, gs):
        # 合并子图的特征
        unbatched_gs = dgl.unbatch(gs)
        # 获取每个子图的节点数
        num_nodes_list = [sub_g.num_nodes() for sub_g in unbatched_gs]
        # 分解节点特征
        unbatched_hs = [sub_g.ndata['h'] for sub_g in unbatched_gs]
        # 获取最大节点数
        max_num_nodes = max(num_nodes_list)
        # 填充子图的特征
        padded_unbatched_hs = []
        for sub_h in unbatched_hs:
            pad_size = max_num_nodes - sub_h.size(0)
            padded_sub_h = torch.cat(
                (sub_h, torch.zeros(size=(pad_size, sub_h.size(1)),
                                    requires_grad=sub_h.requires_grad,
                                    device=sub_h.device)), dim=0)
            padded_unbatched_hs.append(padded_sub_h)
            # 将扩充后的特征添加到列表中
        padded_features = torch.stack(padded_unbatched_hs, dim=0)
        return padded_features