import torch.nn as nn
import torch
# import 
from lib.utils import masked_mae_loss
from model.aug import (
    aug_topology, 
    aug_traffic, 
)

from model.layers import (
    fullyAttentiveEncoder,
    STEncoder, 
    SpatialHeteroModel, 
    TemporalHeteroModel, 
    MLP,    actuallyMLP,
    self_Attention,
    cross_Attention, 
    PositionWise_cross_Attention,
    PositionwiseFeedForward,
)

import numpy as np
class STSSL(nn.Module):
    def __init__(self, args):
        super(STSSL, self).__init__()
    
        self.attention1 = self_Attention(128, 4)
        self.attention2 = self_Attention(128, 4)

        # self.cross_attention1 = PositionWise_cross_Attention(64, 4)
        # self.cross_attention2 = PositionWise_cross_Attention(64, 4)

        self.ff = PositionwiseFeedForward(d_model=128, d_ff=64*4)
        self.mlp = MLP(2*args.d_model, args.d_output)
        self.mae = masked_mae_loss(mask_value=5.0)
        self.args = args
        graph_init = args.graph_init
        ## attention flags
        self.self_attention_flag = args.self_attention_flag
        self.cross_attention_flag = args.cross_attention_flag
        self.feedforward_flag = args.feedforward_flag
        self.layer_norm_flag = args.layer_norm_flag
        self.additional_sa_flag = args.additional_sa_flag
        self.pos_emb_flag = args.pos_emb_flag

        if graph_init == "eye":
            adj = "data/NYCTaxi/adj_mx.npz"
            adj = np.load(adj)["adj_mx"]
            self.learnable_graph = nn.Parameter(torch.eye(adj.shape[1]).float(), requires_grad=False)
        elif graph_init == "zeros":             ## eye sconv because of cheb approximation. actual adj will be: [eye, zero, zero]
            adj = "data/NYCTaxi/adj_mx.npz"
            adj = np.load(adj)["adj_mx"]
            self.learnable_graph = nn.Parameter(torch.zeros_like(torch.tensor(adj).float()), requires_grad=False)
        elif graph_init == "no_sconv":             ## no sconv because of sconv flag, but still need to set some placeholder value to run.
            adj = "data/NYCTaxi/adj_mx.npz"
            adj = np.load(adj)["adj_mx"]
            self.learnable_graph = nn.Parameter(torch.zeros_like(torch.tensor(adj).float()), requires_grad=False)
        elif graph_init == "ones":
            adj = "data/NYCTaxi/adj_mx.npz"
            adj = np.load(adj)["adj_mx"]
            self.learnable_graph = nn.Parameter(torch.ones_like(torch.tensor(adj).float()), requires_grad=False)
        elif graph_init == "random":
            adj = "data/NYCTaxi/adj_mx.npz"
            adj = np.load(adj)["adj_mx"]
            self.learnable_graph = nn.Parameter(torch.empty_like(torch.tensor(adj).float()), requires_grad=False)
            self.xavier_uniform_init(self.learnable_graph)

        elif graph_init == "low_rank" and args.learnable_flag == True:
            r = int(args.num_nodes * (2/3))
            # r = int(args.rank)
            num_low_rank_matrices = 3

            self.matrices1 = nn.Parameter(torch.randn(num_low_rank_matrices, args.num_nodes, r), requires_grad=True).to(args.device)
            self.matrices2 = nn.Parameter(torch.randn(num_low_rank_matrices, args.num_nodes, r), requires_grad=True).to(args.device)
            # self.learnable_graph = nn.Parameter(torch.randn(num_low_rank_matrices, args.num_nodes, args.num_nodes), requires_grad=True).to(args.device)
            # self.learnable_graph = torch.matmul(self.matrices1, self.matrices2.transpose(1, 2))


        elif graph_init == "8_neighbours" and args.learnable_flag == True:
            adj = np.expand_dims(adj, axis=0)  
            adj = np.repeat(adj, 3, axis=0)
            self.learnable_graph = nn.Parameter(torch.from_numpy(adj).float(), requires_grad=True)
        
        elif graph_init == "eye":
            adj = f"data/{args.dataset}/adj_mx.npz"
            adj = np.load(adj)["adj_mx"]
            self.learnable_graph = nn.Parameter(torch.eye(adj.shape[1]).float(), requires_grad=False)
        
        elif graph_init == "8_neighbours" and args.learnable_flag == False:
            adj = f"data/{args.dataset}/adj_mx.npz"
            adj = np.load(adj)["adj_mx"]
            self.learnable_graph = nn.Parameter(torch.from_numpy(adj).float(), requires_grad=False)
        
        elif graph_init == "shared_lpe_T" and args.learnable_flag == False:
            adj = f"data/{args.dataset}/adj_lpe_shared_T.npz"
            adj = np.load(adj)["adj_mx"]
            self.learnable_graph = nn.Parameter(torch.from_numpy(adj).float(), requires_grad=False)

        elif graph_init == "shared_lpe_T_T" and args.learnable_flag == False:
            adj = f"data/{args.dataset}/adj_lpe_shared_T_T.npz"
            adj = np.load(adj)["adj_mx"]
            self.learnable_graph = nn.Parameter(torch.from_numpy(adj).float(), requires_grad=False)

        elif graph_init == "shared_lpe_raw" and args.learnable_flag == False:
            adj = f"data/{args.dataset}/adj_lpe_shared_raw.npz"
            adj = np.load(adj)["adj_mx"]
            self.learnable_graph = nn.Parameter(torch.from_numpy(adj).float(), requires_grad=False)
        
        elif graph_init == "both" and args.learnable_flag == False:
            adj_pt = f"data/{args.dataset}/adj_lpe_shared_raw.npz"
            adj_pt = np.load(adj_pt)["adj_mx"]
            adj_pt = np.expand_dims(adj_pt, axis=0)
            adj_8 = f"data/{args.dataset}/adj_mx.npz"
            adj_8 = np.load(adj_8)["adj_mx"]
            adj_8 = np.expand_dims(adj_8, axis=0)
            adj = np.concatenate((adj_8, adj_pt), axis=0)
            self.learnable_graph = nn.Parameter(torch.from_numpy(adj).float(), requires_grad=False)

        self.dataset = args.dataset

        
        self.encoderA = STEncoder(Kt=3, Ks=args.cheb_order, blocks=[[2, int(args.d_model//2), args.d_model], [args.d_model, int(args.d_model//2), args.d_model]], 
                        input_length=args.input_length, num_nodes=args.num_nodes, droprate=args.dropout, graph_init=graph_init, learnable_flag=args.learnable_flag, row=args.row, col=args.col)
        self.encoderB = STEncoder(Kt=3, Ks=args.cheb_order, blocks=[[2, int(args.d_model//2), args.d_model], [args.d_model, int(args.d_model//2), args.d_model]], 
                        input_length=args.input_length, num_nodes=args.num_nodes, droprate=args.dropout, graph_init=graph_init, learnable_flag=args.learnable_flag, row=args.row, col=args.col)         
        
        
        
        # ## norms
        self.layernorm1 = nn.LayerNorm(128)
        self.layernorm2 = nn.LayerNorm(128)
        self.layernorm3 = nn.LayerNorm(128)
        

        self.dataset = args.dataset
        self.row = args.row
        self.col = args.col
        self.add_8_neighbours = args.add_8
        self.add_eye = args.add_eye

        neighbours = f"data/{args.dataset}/adj_mx.npz"
        neighbours = np.load(neighbours)["adj_mx"]
        self.neighbours = nn.Parameter(torch.from_numpy(neighbours).float(), requires_grad=False).to(self.args.device)
        self.eye = torch.eye(args.num_nodes).to(self.args.device)
        
        self.add_x_encoder = args.add_x_encoder

        from model.vision_transformer import VisionTransformer
        self.encoder = VisionTransformer(
        img_size=(self.args.row, self.args.col),
        patch_size=1,
        in_chans=args.input_length*2,
        embed_dim=64,
        predictor_embed_dim=None,
        depth=4,
        predictor_depth=None,
        num_heads=4,
        mlp_ratio=4,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.4,
        attn_drop_rate=0.4,
        drop_path_rate=0.3,
        norm_layer=torch.nn.LayerNorm,
        init_std=0.02
        )
        self.encoder = self.encoder.to(self.args.device)
        r_path = fr"D:\omer\ST-SSL\logs\{self.dataset}_lpe_shared\jepa-latest.pth.tar"
        checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
        pretrained_dict = checkpoint['encoder']
        msg = self.encoder.load_state_dict(pretrained_dict)
        print(f"msg: {msg}")
        self.freeze_encoder = self.args.freeze_encoder
        if self.freeze_encoder:
            print("Freezing encoder")
            for param in self.encoder.parameters():
                param.requires_grad = False


    def xavier_uniform_init(self, tensor):
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
        std = np.sqrt(2.0 / (fan_in + fan_out))
        nn.init.uniform_(tensor, -std, std) 

    def threshold_top(self, tensor, top_n=8):
        top_values, top_indices = torch.topk(tensor, k=top_n, dim=2, largest=True, sorted=False)
        mask = torch.zeros_like(tensor, dtype=torch.float32)
        mask.scatter_(1, top_indices, True).detach()
        # thresholded_tensor = torch.where(mask, tensor, torch.zeros_like(tensor))
        # thresholded_tensor = torch.where(mask, torch.ones_like(tensor), torch.zeros_like(tensor))
        thresholded_tensor = mask*tensor
        return thresholded_tensor

    def threshold_top_values(self, tensor):
        mask = torch.zeros_like(tensor).detach()
        
        for i in range(tensor.size(0)):
            top_values, top_indices = tensor[i].topk(8, dim=1, largest=True, sorted=False)
            mask[i].scatter_(1, top_indices, 1)
        
        return mask#*tensor
    
    def threshold_top_values_ste(self, tensor):
        mask = torch.zeros_like(tensor).detach()
        
        for i in range(tensor.size(0)):
            top_values, top_indices = tensor[i].topk(8, dim=1, largest=True, sorted=False)
            mask[i].scatter_(1, top_indices, 1)
        # Forward pass: hard thresholding
        thresholded_tensor = mask

        # Hook to modify the gradient during the backward pass: implement STE
        thresholded_tensor = (thresholded_tensor - tensor).detach() + tensor
        return thresholded_tensor
    
    def smooth_threshold_top_values(self, tensor, top_k=8, scale=10):
        thresholded_tensor = torch.zeros_like(tensor)
        for i in range(tensor.size(0)):
            values, _ = tensor[i].topk(top_k, dim=1, largest=True, sorted=True)
            min_top_value = values[:, -1:]  # Minimum value among the top-k values   
            thresholded_tensor[i] = torch.sigmoid(scale * (tensor[i] - min_top_value))
        return thresholded_tensor

    torch.autograd.set_detect_anomaly(True)
    def forward(self, view1, graph):
        print("view1.shape: ", view1.shape)
        if self.dataset == "NYCBike1":
            view1A = view1[:, -4:19, :, :]
            view1B = view1[:, -9:-4, :, :]
        elif self.dataset == "NYCBike2" or self.dataset == "NYCTaxi" or self.dataset == "BJTaxi": 
            view1A = view1[:, -8:35, :, :]
            view1B = view1[:, -17:-8, :, :]
        view1A = view1A.to(self.args.device)
        view1B = view1B.to(self.args.device)

        """get learnable_graph from jepa block here"""
        """lets try just putting the pretrained encoders here and passing the inputs through them to get attention map for every time step"""
        
        view1 = view1    ## n,l,v,c
        B, T, N, D = view1.size()
        view1 = view1.transpose(1, 2).reshape(B, N, -1)
        B, N, D = view1.size()

        view1 = view1.view(B, self.args.row, self.args.col, D).to(self.args.device)
        view1 = view1.permute(0, 3, 1, 2)  # [B, D, R, C]
        B, D, R, C = view1.size()
        masks_enc = torch.ones(B, 1, R*C, dtype=torch.uint8)
        x_encoder, _, attn_list, _, _, _ = self.encoder(view1, masks=masks_enc, pe=None)  ## VisionTransformer
        # print("attn_list: ", len(attn_list))
        # print("attn_list[0].shape: ", attn_list[0].shape)
        ## apply softmax to each attention matrix
        attn_list = [attn.softmax(dim=-1) for attn in attn_list]
        attn_list = torch.stack(attn_list)  # Stack the matrices along a new dimension
        # print("attn_list.shape: ", attn_list.shape)
        avg_attn = torch.mean(attn_list, dim=0)
        # print("avg_attn.shape: ", avg_attn.shape)
        avg_attn = torch.mean(avg_attn, dim=1)
        # print("avg_attn.shape: ", avg_attn.shape)       ## [32, 200, 200]
        import matplotlib.pyplot as plt
        # avg_attn = attn_list[0][:, 0, ...]
        # print("avg_attn.shape: ", avg_attn.shape)       ## [32, 200, 200]
        
        # plt.imshow(avg_attn[0, :, :].detach().cpu().numpy())
        # plt.show()
        
        avg_attn = self.threshold_top_values_ste(avg_attn)
        # plt.imshow(avg_attn[0, :, :].detach().cpu().numpy())
        # plt.show()
        
        if self.add_8_neighbours: 
            avg_attn += self.neighbours
        if self.add_eye:
            avg_attn += self.eye
        # import matplotlib.pyplot as plt
        # plt.imshow(avg_attn[0, :, :].detach().cpu().numpy())
        # plt.show()
        # plt.imshow(avg_attn[1, :, :].detach().cpu().numpy())
        # plt.show()
        # plt.imshow(avg_attn[2, :, :].detach().cpu().numpy())
        # plt.show()
        # plt.imshow(avg_attn[3, :, :].detach().cpu().numpy())
        # plt.show()
        # plt.imshow(avg_attn[4, :, :].detach().cpu().numpy())
        # plt.show()
        # plt.imshow(avg_attn[5, :, :].detach().cpu().numpy())
        # plt.show()
        # plt.imshow(avg_attn[6, :, :].detach().cpu().numpy())
        # plt.show()
        # plt.imshow(avg_attn[7, :, :].detach().cpu().numpy())
        # plt.show()
        # # #avg_attn.shape: [32, 200, 200]   ## one attention map for each sample in the batch
        """ end here """
        
        learnable_graph = avg_attn   ## make 1st channel dimension for einsum to properly message pass
        
        # learnable_graph = avg_attn[0, :, :].unsqueeze(0)  ## for single sample

        # learnable_graph = self.learnable_graph.unsqueeze(0)  
        # B, _, _, _ = view1A.size()
        # learnable_graph = learnable_graph.repeat(B, 1, 1, 1)

        # print(f"view1A.shape: {view1A.shape}, x_encoder.shape: {x_encoder.shape}")
        ##  x_encoder.shape: torch.Size([32, 200, 64]) 64 dim represenation for each node that is spatially aware of surrounding nodes. Lets add this to the output of encoder


        """ check einsum implementation for message passing, is running but probly wrong """
        repr1A = self.encoderA(view1A, learnable_graph) # view1: n,l,v,c; graph: v,v 
        repr1B = self.encoderB(view1B, learnable_graph) # view1: n,l,v,c; graph: v,v 
        # print(f"repr1A.shape: {repr1A.shape}, repr1B.shape: {repr1B.shape}")
        if self.add_x_encoder:
            repr1A += x_encoder.unsqueeze(1)
            repr1B += x_encoder.unsqueeze(1)
        combined_repr = torch.cat((repr1A, repr1B), dim=3)            ## combine along the channel dimension d_model
        
        
        if self.self_attention_flag:
            combined_repr = combined_repr.squeeze(1)
            if self.feedforward_flag:
                combined_repr_copy = combined_repr
                combined_repr = self.ff(combined_repr)
                combined_repr = combined_repr + combined_repr_copy
                if self.layer_norm_flag == True:
                    combined_repr = self.layernorm1(combined_repr)
            # combined_repr_copy = combined_repr
            # combined_repr = self.ffA1(combined_repr)   ## for inter channel mixing
            # combined_repr = combined_repr + combined_repr_copy
            çombined_repr_copy = combined_repr
            combined_repr = self.attention1(combined_repr)
            combined_repr = combined_repr + çombined_repr_copy  # skip connection
            if self.layer_norm_flag == True:
                combined_repr = self.layernorm2(combined_repr)
            combined_repr_copy = combined_repr
            combined_repr = self.attention2(combined_repr)
            combined_repr = combined_repr + combined_repr_copy  # skip connection
            if self.layer_norm_flag == True:
                combined_repr = self.layernorm3(combined_repr)
            combined_repr = combined_repr.unsqueeze(1)

        repr2 = None
        learnable_graph = None
        return combined_repr, learnable_graph

    def fetch_spatial_sim(self):
        """
        Fetch the region similarity matrix generated by region embedding.
        Note this can be called only when spatial_sim is True.
        :return sim_mx: tensor, similarity matrix, (v, v)
        """
        return self.encoder.s_sim_mx.cpu()
    
    def fetch_temporal_sim(self):
        return self.encoder.t_sim_mx.cpu()

    def predict(self, z1, z2):
        '''Predicting future traffic flow.
        :param z1, z2 (tensor): shape nvc
        :return: nlvc, l=1, c=2
        '''
        # print("z1.shape: ", z1.shape)
        return self.mlp(z1)

    def loss(self, z1, z2, y_true, scaler, loss_weights):
        l1 = self.pred_loss(z1, z2, y_true, scaler)
        sep_loss = [l1.item()]
        loss = loss_weights[0] * l1 

        # l2 = self.temporal_loss(z1, z2)
        # sep_loss.append(l2.item())
        # if self.args.T_Loss==1:
            # loss += loss_weights[1] * l2
        
        # l3 = self.spatial_loss(z1, z2)
        # sep_loss.append(l3.item())
        # if self.args.S_Loss==1:
            # print("spatial loss: ", l3)
            # loss += loss_weights[2] * l3 
        # print("predLoss: ", l1.item(), "temporalLoss: ", l2.item(), "spatialLoss: ", l3.item())
        return loss, sep_loss

    def pred_loss(self, z1, z2, y_true, scaler):
        y_pred = scaler.inverse_transform(self.predict(z1, z2))
        y_true = scaler.inverse_transform(y_true)
 
        loss = self.args.yita * self.mae(y_pred[..., 0], y_true[..., 0]) + \
                (1 - self.args.yita) * self.mae(y_pred[..., 1], y_true[..., 1])
        return loss
    
    def temporal_loss(self, z1, z2):
        return self.thm(z1, z2)

    def spatial_loss(self, z1, z2):
        return self.shm(z1, z2)
    