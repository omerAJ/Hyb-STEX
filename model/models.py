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

from model.vision_transformer_utils import apply_masks_targets
import torch.nn.functional as F
import numpy as np
class STSSL(nn.Module):
    def __init__(self, args):
        super(STSSL, self).__init__()
    
        self.attention1 = self_Attention(128, 4)
        self.attention2 = self_Attention(128, 4)

        
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
        
        
        if graph_init == "8_neighbours" and args.learnable_flag == False:
            adj = f"data/{args.dataset}/adj_mx.npz"
            adj = np.load(adj)["adj_mx"]
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

        """for jepa"""
        from model.vision_transformer import VisionTransformer, VisionTransformerPredictor
        self.encoder = VisionTransformer(
        img_size=(self.args.row, self.args.col),
        patch_size=1,
        in_chans=2,
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
        # self.predictor = VisionTransformerPredictor(
        # img_size=(args.row, args.col),
        # embed_dim=64,
        # predictor_embed_dim=64//2,
        # depth=2,
        # num_heads=4,
        # mlp_ratio=4,
        # qkv_bias=False,
        # qk_scale=None,
        # drop_rate=0.4,
        # attn_drop_rate=0.4,
        # drop_path_rate=0.3,
        # norm_layer=torch.nn.LayerNorm,
        # init_std=0.02
        # )

        # import copy
        # self.target_encoder = copy.deepcopy(self.encoder)

        self.encoder = self.encoder.to(self.args.device)
        # self.predictor = self.predictor.to(self.args.device)
        # self.target_encoder = self.target_encoder.to(self.args.device)

        r_path = fr"D:\omer\ST-SSL\logs\{self.dataset}_individualT\jepa-latest.pth.tar"
        
        checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
        pretrained_dict = checkpoint['encoder']
        msg = self.encoder.load_state_dict(pretrained_dict)
        print(f"encoder msg: {msg}")
        
        # pretrained_dict = checkpoint['predictor']
        # msg = self.predictor.load_state_dict(pretrained_dict)
        # print(f"predictor msg: {msg}")
        
        # pretrained_dict = checkpoint['target_encoder']
        # msg = self.target_encoder.load_state_dict(pretrained_dict)
        # print(f"target_encoder msg: {msg}")

        # for p in self.target_encoder.parameters():
        #     p.requires_grad = False
        
        self.freeze_encoder = self.args.freeze_encoder
        if self.freeze_encoder:
            print("Freezing encoder")
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        T = 8
        self.weights = nn.Parameter(torch.ones(T) / T)
        # initialize the weights with a uniform distribution
        # nn.init.normal_(self.weights, mean=1/T, std=0.02)
        

    def xavier_uniform_init(self, tensor):
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
        std = np.sqrt(2.0 / (fan_in + fan_out))
        nn.init.uniform_(tensor, -std, std) 

    
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
    
    def threshold_top_values_ste_PosNeg(self, tensor):
        mask = torch.zeros_like(tensor).detach()
        
        for i in range(tensor.size(0)):
            # Get the top 8 positive values
            top_pos_values, top_pos_indices = tensor[i].topk(8, dim=1, largest=True, sorted=False)
            mask[i].scatter_(1, top_pos_indices, 1)

            # Get the top 8 negative values
            top_neg_values, top_neg_indices = tensor[i].topk(8, dim=1, largest=False, sorted=False)
            mask[i].scatter_(1, top_neg_indices, -1)

        # Forward pass: hard thresholding
        thresholded_tensor = mask

        # Hook to modify the gradient during the backward pass: implement STE
        thresholded_tensor = (thresholded_tensor - tensor).detach() + tensor
        return thresholded_tensor
    
    torch.autograd.set_detect_anomaly(True)
    
    def generateMasks(self, data):
        # :param data: tensor of shape [B, R, C, D]
        # :returns: (data, 1x masks_enc, 4x masks_pred)
        import numpy as np
        B, R, C, D = data.size()

        masks_enc = torch.zeros(B, R, C, dtype=torch.uint8)
        masks_pred = torch.zeros(4, B, R, C, dtype=torch.uint8)

        
        masks_enc = masks_enc.flatten(1)
        masks_pred = masks_pred.flatten(2)
        ctxt_size = torch.randint(50, 170, (1,)).item()       ## low (inclusive), high (exclusive)
        leftOutNodes = R*C - ctxt_size
        trgt_size = torch.randint(2*leftOutNodes//3, leftOutNodes, (1,)).item()  
        _try=0
        for b in range(B):
            ctxt_indices = np.random.choice(R*C, size=ctxt_size, replace=False)
            available_indices = np.setdiff1d(np.arange(R*C), ctxt_indices)
            masks_enc[b, ctxt_indices] = 1 
            for i in range(4):
                trgt_indices = np.random.choice(available_indices, size=trgt_size, replace=False)
                masks_pred[i, b, trgt_indices] = 1
        masks_enc = masks_enc.view(B, 1, R*C)
        masks_pred = masks_pred.view(4, B, R*C)
        return (data, masks_enc, masks_pred.transpose(0, 1))
    
    
    def forward(self, view1, graph):
        
        if self.dataset == "NYCBike1":
            view1A = view1[:, -4:19, :, :]
            view1B = view1[:, -9:-4, :, :]
            view1 = view1[:, -4:19, :, :]
        elif self.dataset == "NYCBike2" or self.dataset == "NYCTaxi" or self.dataset == "BJTaxi": 
            view1A = view1[:, -8:35, :, :]
            view1B = view1[:, -17:-8, :, :]
            view1 = view1[:, -8:35, :, :]
        view1A = view1A.to(self.args.device)
        view1B = view1B.to(self.args.device)
        # print(f"view1.shape: {view1.shape}, view1A.shape: {view1A.shape}, view1B.shape: {view1B.shape}")  ## view1.shape: torch.Size([32, 17, 200, 2]), view1A.shape: torch.Size([32, 8, 200, 2]), view1B.shape: torch.Size([32, 9, 200, 2])
        """get learnable_graph from jepa block here"""
        """lets try just putting the pretrained encoders here and passing the inputs through them to get attention map for every time step"""
        
        B, T, N, D = view1.size()
        # z_list = []
        # h_list = []
        # for t in range(T):
        #     _view1 = view1[:, t, :, :].unsqueeze(1)
        #     B, _, N, D = _view1.size()
        #     _view1 = _view1.transpose(1, 2).reshape(B, N, -1)
        #     B, N, D = _view1.size()

        #     _view1 = _view1.view(B, self.args.row, self.args.col, D).to(self.args.device)
        #     B, R, C, D = _view1.size()
        
        #     _view1 = self.generateMasks(_view1)
        #     imgs, masks_enc, masks_pred = _view1
            
        #     imgs = imgs.permute(0, 3, 1, 2)  ## [B, R, C, D] -> [B, D, R, C]
        #     masks_pred = masks_pred.flatten(2) ## [B, 4, R, C] -> [B, 4, R*C]
        #     masks_enc = masks_enc.flatten(1).unsqueeze(1) ## [B, R, C] -> [B, 1, R*C]
        
        #     z, pe, attn_list = self.encoder(imgs, masks=masks_enc, pe=None)  ## VisionTransformer
        #     z = self.predictor(x=z, masks_enc=masks_enc, masks_pred=masks_pred, pe=pe)   ## VisionTransformerPredictor
        #     with torch.no_grad():
        #         h = self.target_encoder(imgs, masks=None, pe=pe)
        #         h = F.layer_norm(h, (h.size(-1),))
        #         h = apply_masks_targets(h, masks_pred)
        #     # print(f"t: {t}, z.shape: {z.shape}, h.shape: {h.shape}")
        #     z_list.append(z)
        #     h_list.append(h)
        # z = torch.cat(z_list, dim=1)  # Stack along the time dimension
        # h = torch.cat(h_list, dim=1)
        
        
        # import matplotlib.pyplot as plt
        
        
        
        """ end here """
        B, T, N, D = view1.size()
        ## run a forward through the encoder once again, this time with no mask to get the full att_mx
        masks_enc = torch.ones(B, 1, 200, dtype=torch.uint8)
        
        # Apply softmax to the weights
        normalized_weights = torch.softmax(self.weights, dim=0)
        # normalized_weights = self.weights
        avg_attn_accum = torch.zeros(B, N, N, device=self.args.device)
        # for t in range(T):
        for t in range(T):
            _view1 = view1[:, t, :, :].unsqueeze(1)
            B, _, N, D = _view1.size()
            _view1 = _view1.transpose(1, 2).reshape(B, N, -1)
            B, N, D = _view1.size()
            _view1 = _view1.view(B, self.args.row, self.args.col, D).to(self.args.device)
            B, R, C, D = _view1.size()  
            _view1 = _view1.permute(0, 3, 1, 2)  ## [B, R, C, D] -> [B, D, R, C]
            _, _, attn_list = self.encoder(_view1, masks=masks_enc, pe=None)  ## VisionTransformer
            
            # attn_list = [attn.softmax(dim=-1) for attn in attn_list]
            
            attn_list = torch.stack(attn_list)  # Stack the matrices along a new dimension
            ## take softmax here instead of above in the list comprehension
            # print(f"t: {t}, attn_list.shape: {attn_list.shape}")
            avg_attn = torch.mean(attn_list, dim=(0, 2))
            # avg_attn = torch.mean(avg_attn, dim=1)  ## included 1 in the above mean, over there it is dim=2

            avg_attn = self.threshold_top_values_ste_PosNeg(avg_attn)

            avg_attn_accum += normalized_weights[t] * avg_attn  # Weighted (learnable) accumulation
            # print(f"t: {t}, avg_attn norm: {np.linalg.norm(avg_attn.cpu().detach().numpy())}")

        if self.add_8_neighbours: 
            avg_attn_accum += self.neighbours
        if self.add_eye:
            avg_attn_accum += self.eye    
        learnable_graph = avg_attn_accum   ## make 1st channel dimension for einsum to properly message pass
            
        """ check einsum implementation for message passing, is running but probly wrong """
        repr1A = self.encoderA(view1A, learnable_graph) # view1: n,l,v,c; graph: v,v 
        repr1B = self.encoderB(view1B, learnable_graph) # view1: n,l,v,c; graph: v,v 
        # print(f"repr1A.shape: {repr1A.shape}, repr1B.shape: {repr1B.shape}")
        
        """
        if self.add_x_encoder:
            repr1A += x_encoder.unsqueeze(1)
            repr1B += x_encoder.unsqueeze(1)
        """

        combined_repr = torch.cat((repr1A, repr1B), dim=3)            ## combine along the channel dimension d_model
        
        
        if self.self_attention_flag:
            combined_repr = combined_repr.squeeze(1)
            if self.feedforward_flag:
                combined_repr_copy = combined_repr
                combined_repr = self.ff(combined_repr)
                combined_repr = combined_repr + combined_repr_copy
                if self.layer_norm_flag == True:
                    combined_repr = self.layernorm1(combined_repr)
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
        z = None
        h = None
        return combined_repr, learnable_graph, z, h

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

    def loss(self, z1, z2, y_true, scaler, loss_weights, z, h):
        l1 = self.pred_loss(z1, z2, y_true, scaler)
        # sep_loss = [l1.item()]
        loss = l1 

        # loss_jepa = F.smooth_l1_loss(z, h)
        # loss_l2 = F.mse_loss(z, h)
        # loss_jepa += loss_l2
        # sep_loss = loss_jepa.item()
        # loss += loss_jepa
        sep_loss=loss.item()
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
    