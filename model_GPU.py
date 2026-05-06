import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv

EPS = torch.tensor(1e-15)


                                                                                  

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
                                      
                                     

                                       
        target = target.float()

                   
        epsilon = 1e-7
        pred = torch.clamp(pred, epsilon, 1. - epsilon)

        bce_loss = - target * torch.log(pred) - (1 - target) * torch.log(1 - pred)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


                                                                        

class BilinearDecoder(nn.Module):
    def __init__(self, feature_dim, dropout=0.5):
        super(BilinearDecoder, self).__init__()
        self.weights = nn.Parameter(torch.Tensor(feature_dim, feature_dim))
        nn.init.xavier_uniform_(self.weights)
                                              
        self.layernorm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, circ_emb, dis_emb):
                           
        circ = self.dropout(self.layernorm(circ_emb))
        dis = self.dropout(self.layernorm(dis_emb))

                      
        inter = torch.mm(circ, self.weights)
        bilinear_logits = (inter * dis).sum(1)

                               
        res_logits = (circ * dis).sum(1)

                                                                                
        return bilinear_logits + res_logits



class Hgnn(nn.Module):
    def __init__(self, in_channels, dim_1):
        super(Hgnn, self).__init__()
        self.conv1 = HypergraphConv(in_channels, dim_1)
        self.conv2 = HypergraphConv(dim_1, dim_1 // 4)

                                                              

    def forward(self, x, edge, hyperedge_weight=None):
        x = torch.relu(self.conv1(x, edge, hyperedge_weight=hyperedge_weight))
        x = torch.relu(self.conv2(x, edge, hyperedge_weight=hyperedge_weight))
        return x



class HgnnEncoder(nn.Module):




    def __init__(self, in_channels, dim_1, mi_edge_drop=0.2, do_edge_drop=0.2, dual_ablation: str = "full", view_drop: float = 0.0, use_view_gate: bool = False, view_fusion: str = "attn"):

        super(HgnnEncoder, self).__init__()

                                   
        self.conv = Hgnn(in_channels, dim_1)
        self.conv_mi = Hgnn(in_channels, dim_1)

                                         
        self.g_mi = nn.Parameter(torch.tensor(-2.0))
        self.mi_edge_drop = mi_edge_drop
                                          
        self.conv_do = Hgnn(in_channels, dim_1)
        self.g_do = nn.Parameter(torch.tensor(-2.0))
        self.do_edge_drop = do_edge_drop
                                                             
        self.use_view_gate = bool(use_view_gate)
        self.view_drop = float(view_drop)

        self.fusion_attention_c = FusionAttention(dim_1 // 4)
        self.fusion_attention_d = FusionAttention(dim_1 // 4)
        self.dual_ablation = str(dual_ablation)
        self.view_fusion = str(view_fusion).lower().strip()
        self.dual_ablation = dual_ablation
                                   
                                                   
                                   
                                              

                                                        
                                                        
                                                              
        self.view_gate_c = nn.Parameter(torch.zeros(5))
        self.view_gate_d = nn.Parameter(torch.zeros(5))



    def generate_hye_emb(self, node_embedding, hye_len, hye_node):




        node_embedding = node_embedding.to(node_embedding.device)

        zeros = torch.zeros(1, node_embedding.shape[1], device=node_embedding.device)
        node_embedding = torch.cat([zeros, node_embedding], dim=0)

        if not torch.is_tensor(hye_len):
            hye_len = torch.tensor(
                hye_len, dtype=torch.float32, device=node_embedding.device
            )
        else:
            hye_len = hye_len.to(node_embedding.device)

        if hye_len.dim() == 1:
            hye_len = hye_len.view(-1, 1)

        hye_node = hye_node.to(node_embedding.device)

        max_valid = node_embedding.shape[0] - 1
        min_valid = 0

        cur_min = hye_node.min().item()
        cur_max = hye_node.max().item()

        if cur_min < min_valid or cur_max > max_valid:









            hye_node = hye_node.clone()
            hye_node[hye_node < min_valid] = 0
            hye_node[hye_node > max_valid] = 0

        seq_h = node_embedding[hye_node]
        eps_device = EPS.to(node_embedding.device)
        hs = torch.sum(seq_h, dim=1) / (hye_len + eps_device)
        return hs

    def _edge_dropout(self, edge_index: torch.Tensor, drop_prob: float) -> torch.Tensor:

        if (not self.training) or drop_prob <= 0:
            return edge_index
        if (not torch.is_tensor(edge_index)) or edge_index.numel() == 0:
            return edge_index

        m = edge_index.size(1)
        keep = torch.rand(m, device=edge_index.device) > drop_prob
                            
        if keep.sum().item() == 0:
            keep[torch.randint(0, m, (1,), device=edge_index.device)] = True
        return edge_index[:, keep]

    def _apply_view_regularization(self, x_stack: torch.Tensor, g_view: torch.Tensor) -> torch.Tensor:




                                    
        if self.use_view_gate:
            gate = 2.0 * torch.sigmoid(g_view).view(1, -1, 1)                 
            x_stack = x_stack * gate

                                             
        if self.training and self.view_drop > 0:
            V = x_stack.size(1)
            mask = (torch.rand(V, device=x_stack.device) > self.view_drop).float()       
                   
            if mask.sum().item() == 0:
                mask[torch.randint(0, V, (1,), device=x_stack.device)] = 1.0

                                 
            scale = mask.mean().clamp_min(1e-12)
            x_stack = x_stack * mask.view(1, -1, 1) / scale

        return x_stack


    def forward(self, x_ls, hyg_ls):












        if hyg_ls is None:
            raise ValueError("HgnnEncoder.forward 收到的 hyg_ls 是 None")

                     
                                              
        if len(hyg_ls) == 4:
            G_d, G_c, G_cc, G_dd = hyg_ls
            G_mc = None
            G_md = None
        elif len(hyg_ls) == 5:
            G_d, G_c, G_cc, G_dd, G_mc = hyg_ls
            G_md = None
        elif len(hyg_ls) == 6:
            G_d, G_c, G_cc, G_dd, G_mc, G_md = hyg_ls
        else:
            raise ValueError(f"hyg_ls 长度应为 4/5/6，但收到 {len(hyg_ls)}")

                                            
        x1_cm = self.conv(x_ls[0], G_d[0])              
        x1_mc = self.conv(x_ls[1], G_c[0])           
        x1_cc = self.conv(x_ls[2], G_cc[0])           
        x1_mm = self.conv(x_ls[3], G_dd[0])              

                                         
        x2_cm = self.generate_hye_emb(x1_cm, G_d[1], G_d[2])                      
        x2_mc = self.generate_hye_emb(x1_mc, G_c[1], G_c[2])                         
        x2_cc = self.generate_hye_emb(x1_cc, G_cc[1], G_cc[2])                      
        x2_mm = self.generate_hye_emb(x1_mm, G_dd[1], G_dd[2])                         
                                                                             
             
                                                  
                                               
        if self.dual_ablation != "full":
            if self.dual_ablation == "node_only":
                                     
                x2_cm = torch.zeros_like(x2_cm)
                x2_mc = torch.zeros_like(x2_mc)
                x2_cc = torch.zeros_like(x2_cc)
                x2_mm = torch.zeros_like(x2_mm)

            elif self.dual_ablation == "hyper_only":
                                                           
                x1_cm = torch.zeros_like(x1_cm)
                x1_mc = torch.zeros_like(x1_mc)
                x1_cc = torch.zeros_like(x1_cc)
                x1_mm = torch.zeros_like(x1_mm)

            elif self.dual_ablation == "bipartite_x2_only":
                                                                
                x2_cc = torch.zeros_like(x2_cc)
                x2_mm = torch.zeros_like(x2_mm)

            elif self.dual_ablation == "similarity_x2_only":
                                                                
                x2_cm = torch.zeros_like(x2_cm)
                x2_mc = torch.zeros_like(x2_mc)

            else:
                raise ValueError(f"Unknown dual_ablation: {self.dual_ablation}")
                                                                             


                         
        n_circ = x_ls[1].size(0)            
        m_dis = x_ls[0].size(0)               

        def _pad_to(x, target_len, name):
            n = x.size(0)
            if n == target_len:
                return x
            if n > target_len:
                                                                                 
                return x[:target_len, :]
            else:
                                                                                    
                pad = torch.zeros(target_len - n, x.size(1), device=x.device, dtype=x.dtype)
                return torch.cat([x, pad], dim=0)

                                                        
        if G_mc is not None:
                                                             
            if torch.is_tensor(G_mc[0]) and G_mc[0].numel() == 0:
                               
                x1_mi = torch.zeros_like(x1_mc)
            else:
                                              
                edge_mi = self._edge_dropout(G_mc[0], self.mi_edge_drop)

                                                                 
                hye_len_mi = G_mc[1].to(x_ls[1].device)
                if hye_len_mi.dim() > 1:
                    hye_len_mi = hye_len_mi.view(-1)
                hyper_w = torch.log1p(hye_len_mi)
                hyper_w = hyper_w / (hyper_w.mean() + 1e-12)

                                             
                x1_mi_raw = self.conv_mi(x_ls[1], edge_mi, hyperedge_weight=hyper_w)

                             
                x1_mi = torch.sigmoid(self.g_mi) * x1_mi_raw

                                                                  
        else:
            x1_mi = torch.zeros_like(x1_mc)

                                                           
        if G_md is not None:
                                                                
            if torch.is_tensor(G_md[0]) and G_md[0].numel() == 0:
                x1_do = torch.zeros_like(x1_cm)
            else:
                edge_do = self._edge_dropout(G_md[0], self.do_edge_drop)

                hye_len_do = G_md[1]
                if not torch.is_tensor(hye_len_do):
                    hye_len_do = torch.tensor(hye_len_do, device=edge_do.device, dtype=torch.float32)
                else:
                    hye_len_do = hye_len_do.to(edge_do.device).float()
                if hye_len_do.dim() > 1:
                    hye_len_do = hye_len_do.view(-1)

                hyper_w_do = torch.log1p(hye_len_do)
                hyper_w_do = hyper_w_do / (hyper_w_do.mean() + 1e-12)

                x1_do_raw = self.conv_do(x_ls[0], edge_do, hyperedge_weight=hyper_w_do)
                x1_do = torch.sigmoid(self.g_do) * x1_do_raw
        else:
            x1_do = torch.zeros_like(x1_cm)

                                                                  
        x1_mc = _pad_to(x1_mc, n_circ, "x1_mc")
        x2_cm = _pad_to(x2_cm, n_circ, "x2_cm")
        x1_cc = _pad_to(x1_cc, n_circ, "x1_cc")
        x2_cc = _pad_to(x2_cc, n_circ, "x2_cc")
        x1_mi = _pad_to(x1_mi, n_circ, "x1_mi")

                                
        x_c = torch.stack((x1_mc, x2_cm, x1_cc, x2_cc, x1_mi), dim=1)              
        x_c = self._apply_view_regularization(x_c, self.view_gate_c)

                              
        x1_cm = _pad_to(x1_cm, m_dis, "x1_cm")
        x2_mc = _pad_to(x2_mc, m_dis, "x2_mc")
        x1_mm = _pad_to(x1_mm, m_dis, "x1_mm")
        x2_mm = _pad_to(x2_mm, m_dis, "x2_mm")
        x1_do = _pad_to(x1_do, m_dis, "x1_do")
                                           
        x_m = torch.stack((x1_cm, x2_mc, x1_mm, x2_mm, x1_do), dim=1)              
        x_m = self._apply_view_regularization(x_m, self.view_gate_d)

                                                     
                                                  
                                                     
        if self.dual_ablation == "node_only":
            x_c = x_c.clone()
            x_m = x_m.clone()
            x_c[:, 1, :] = 0.0
            x_c[:, 3, :] = 0.0
            x_m[:, 1, :] = 0.0
            x_m[:, 3, :] = 0.0

                                                              
        if self.view_fusion == "mean":
            h_c = x_c.mean(dim=1)
            h_m = x_m.mean(dim=1)
        else:
            h_c = self.fusion_attention_c(x_c)
            h_m = self.fusion_attention_d(x_m)
        return h_c, h_m


class BioEncoder(nn.Module):



    def __init__(self, dim_circ, dim_dis, output):
        super(BioEncoder, self).__init__()

        self.circ_layer1 = nn.Linear(dim_circ, output)
        self.batch_circ1 = nn.BatchNorm1d(output)

        self.dis_layer1 = nn.Linear(dim_dis, output)
        self.batch_dis1 = nn.BatchNorm1d(output)

        self.relu = nn.ReLU()
        self.reset_para()

    def reset_para(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        return

    def forward(self, circ_feat, disease_feat):
        x_circ = self.circ_layer1(circ_feat)
        x_circ = self.batch_circ1(F.relu(x_circ))

        x_dis = self.dis_layer1(disease_feat)
        x_dis = self.batch_dis1(F.relu(x_dis))

        return x_circ, x_dis


class Decoder(nn.Module):
    def __init__(self, in_channels):
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.fc1 = nn.Linear(in_channels, in_channels // 2)
        self.batch1 = nn.BatchNorm1d(in_channels // 2)
        self.fc2 = nn.Linear(in_channels // 2, in_channels // 4)
        self.batch2 = nn.BatchNorm1d(in_channels // 4)
        self.fc23 = nn.Linear(in_channels // 4, in_channels // 8)
        self.batch23 = nn.BatchNorm1d(in_channels // 8)
        self.fc3 = nn.Linear(in_channels // 8, 1)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, Z_c_embed, Z_d_embed, circRNA_id, disease_id):
        h_0 = torch.cat((Z_c_embed[circRNA_id, :], Z_d_embed[disease_id, :]), dim=1)
        h_1 = torch.tanh(self.fc1(h_0))
        h_1 = self.batch1(h_1)
        h_2 = torch.tanh(self.fc2(h_1))
        h_2 = self.batch2(h_2)
        h_23 = torch.tanh(self.fc23(h_2))
        h_23 = self.batch23(h_23)
        h_3 = self.fc3(h_23)
        return h_23, h_3.squeeze(dim=1)

class PairMLPDecoder(nn.Module):




    def __init__(self, d, hidden=256, dropout=0.3):
        super().__init__()
        in_dim = 4 * d                      
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, Z_c_embed, Z_d_embed, c_idx, d_idx):
        c = Z_c_embed[c_idx, :]
        d = Z_d_embed[d_idx, :]
        x = torch.cat([c, d, torch.abs(c - d), c * d], dim=1)
        logits = self.mlp(x).squeeze(1)
        return x, logits


class FusionAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(FusionAttention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)
        beta = torch.softmax(w, dim=0)
        beta = beta.expand((z.shape[0],) + beta.shape)
        return (beta * z).sum(1)


class DHGCDA(torch.nn.Module):



    def __init__(self, bio_encoder, hygraph_encoder, decoder):
        super(DHGCDA, self).__init__()
        self.bio_encoder = bio_encoder
        self.hygraph_encoder = hygraph_encoder
        self.decoder = decoder

    def forward(self, circ_feat, disease_feat, c_idx, d_idx, hy_ls):








        if hy_ls is None:
            raise ValueError(
                "DHGCDA.forward 收到的 hy_ls 是 None，请检查 main_GPU.py 中传入的超图结构。"
            )

                   
        x_circ, x_dis = self.bio_encoder(circ_feat, disease_feat)

                    
        x_ls = [x_dis, x_circ, x_circ, x_dis]

                 
        Z_c_raw, Z_d_raw = self.hygraph_encoder(x_ls, hy_ls)

                   
        Z_c = F.normalize(Z_c_raw, dim=1, p=2)
        Z_d = F.normalize(Z_d_raw, dim=1, p=2)

                               
        z_ij, logits = self.decoder(Z_c, Z_d, c_idx, d_idx)

        return z_ij, logits, Z_c, Z_d
