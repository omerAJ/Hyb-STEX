import torch.nn as nn
import torch
# import 
from lib.utils import masked_mae_loss, masked_mse_loss, masked_gumbell_loss, masked_frechet_loss
# from model.aug import (
#     aug_topology, 
#     aug_traffic, 
# )
import sys
import os

# Get the root directory where your running file is located
root_dir = os.path.dirname(os.path.abspath(__file__))

# Add the root directory to sys.path
sys.path.append(root_dir)

# Now, you can import the required components from layers.py
from layers import (
    STEncoder, 
    MLP,
    self_Attention,
    PositionwiseFeedForward,
    attentive_fusion,
)


# from model.vision_transformer_utils import apply_masks_targets
import torch.nn.functional as F
import numpy as np
class STSSL(nn.Module):
    def __init__(self, args):
        super(STSSL, self).__init__()
        
        # if args.load_path is not None:
        #     import os
        #     import sys
        #     model_dir = os.path.dirname(args.load_path)
        #     sys.path.append(model_dir)
        #     from layers import (
        #         STEncoder, 
        #         MLP,
        #         self_Attention,
        #         PositionwiseFeedForward,
        #         attentive_fusion,
        #     )
        self.args = args

        # self.attention1 = self_Attention(int((2)*args.d_model), 4)
        # self.attention2 = self_Attention(int((2)*args.d_model), 4)
        
        self.attentive_fuse = attentive_fusion(int((2)*args.d_model), n_heads=4, ln=False)
        self.attentive_fuse_cls = attentive_fusion(int((2)*args.d_model), n_heads=4, ln=False)

        self.ff = PositionwiseFeedForward(d_model=128, d_ff=64*4)
        self.mlp = MLP(int((2)*args.d_model), args.d_output)
        self.mlp_cls = MLP(int((2)*args.d_model), args.d_output)
        # self.mlp_bias = MLP(int((2)*args.d_model), args.d_output)
        # self.mlp_bias.fc1.linear.bias.data.fill_(+0.5)  ## bias it to predicting normal
        # self.mlp_bias.fc2.linear.bias.data.fill_(+0.5)  ## bias it to predicting normal
        # self.mlp_cls.fc1.linear.bias.data.fill_(+0.5)  ## bias it to predicting normal
        # self.mlp_cls.fc2.linear.bias.data.fill_(+0.5)  ## bias it to predicting normal
        # self.mlp_classifier.fc2.linear.bias.data.fill_(-1)  ## bias it to predicting normal
        self.loss_fun_val = masked_mae_loss(mask_value=5.0)
        if args.loss == 'mae':
            self.loss_fun = masked_mae_loss(mask_value=5.0)
        elif args.loss == 'mse':
            self.loss_fun = masked_mse_loss(mask_value=5.0)
        elif args.loss == 'gumbell':
            self.loss_fun = masked_gumbell_loss(mask_value=5.0)
        elif args.loss == 'frechet':
            self.loss_fun = masked_frechet_loss(mask_value=5.0)
        self.args = args
        graph_init = args.graph_init
        ## attention flags
        self.self_attention_flag = args.self_attention_flag
        self.cross_attention_flag = args.cross_attention_flag
        self.feedforward_flag = args.feedforward_flag
        self.layer_norm_flag = args.layer_norm_flag
        self.additional_sa_flag = args.additional_sa_flag
        self.pos_emb_flag = args.pos_emb_flag
        self.threshold_adj_mx = args.threshold_adj_mx
        self.dataset = args.dataset
        
        ## A: 2->32->64->64->32->64 
        ## B: 2->16->32->32->16->32 
        self.encoderA = STEncoder(Kt=3, Ks=args.cheb_order, blocks=[[2, int(args.d_model//2), args.d_model], [args.d_model, int(args.d_model//2), args.d_model]], 
                        input_length=args.input_length, num_nodes=args.num_nodes, droprate=args.dropout, graph_init=graph_init, learnable_flag=args.learnable_flag, row=args.row, col=args.col, threshold_adj_mx=args.threshold_adj_mx, do_affinity=args.affinity_conv)
        self.encoderB = STEncoder(Kt=3, Ks=args.cheb_order, blocks=[[2, int(args.d_model//2), args.d_model], [args.d_model, int(args.d_model//2), args.d_model]], 
                        input_length=args.input_length, num_nodes=args.num_nodes, droprate=args.dropout, graph_init=graph_init, learnable_flag=args.learnable_flag, row=args.row, col=args.col, threshold_adj_mx=args.threshold_adj_mx, do_affinity=args.affinity_conv)         
        
        # self.encoderA_cls = STEncoder(Kt=3, Ks=args.cheb_order, blocks=[[2, int(args.d_model//2), args.d_model], [args.d_model, int(args.d_model//2), args.d_model]], 
        #                 input_length=args.input_length, num_nodes=args.num_nodes, droprate=args.dropout, graph_init=graph_init, learnable_flag=args.learnable_flag, row=args.row, col=args.col, threshold_adj_mx=args.threshold_adj_mx, do_affinity=args.affinity_conv)
        # self.encoderB_cls = STEncoder(Kt=3, Ks=args.cheb_order, blocks=[[2, int(args.d_model//2), args.d_model], [args.d_model, int(args.d_model//2), args.d_model]], 
        #                 input_length=args.input_length, num_nodes=args.num_nodes, droprate=args.dropout, graph_init=graph_init, learnable_flag=args.learnable_flag, row=args.row, col=args.col, threshold_adj_mx=args.threshold_adj_mx, do_affinity=args.affinity_conv)         
        
        # ## norms
        self.layernorm1 = nn.LayerNorm(int((2)*args.d_model))
        self.layernorm2 = nn.LayerNorm(int((2)*args.d_model))
        self.layernorm3 = nn.LayerNorm(int((2)*args.d_model))
        

        self.dataset = args.dataset
        self.row = args.row
        self.col = args.col
        self.add_8_neighbours = args.add_8
        self.add_eye = args.add_eye

        neighbours = args.graph_file
        neighbours = np.load(neighbours)["adj_mx"]
        # self.neighbours = nn.Parameter(torch.from_numpy(neighbours).float(), requires_grad=False).to(self.args.device)

        self.neighbours = torch.from_numpy(neighbours).float().to(self.args.device)

        self.eye = torch.eye(args.num_nodes).to(self.args.device)
        
        self.add_x_encoder = args.add_x_encoder

        T = 8
        N = 200
        self.weights = nn.Parameter(torch.ones(N) / N)
        # self.key_projection = nn.Linear(int((2)*args.d_model), int((2)*args.d_model))
        self.ff_key_projection_bias = PositionwiseFeedForward(d_model=128, d_ff=64*4)
        # self.project_to_classify = nn.Linear(int((2)*args.d_model), int((2)*args.d_model))
        self.ff_to_cls = PositionwiseFeedForward(d_model=128, d_ff=128*4)
        # self.learnable_vectors_bias = nn.Parameter(torch.zeros(1, 1, args.num_nodes, 128, 4), requires_grad=True)
        self.learnable_vectors_bias = nn.Parameter(torch.zeros(1, 1, 128, 4), requires_grad=True)
        # self.learnable_bias_bias = nn.Parameter(torch.zeros(1, 1, args.num_nodes, 2), requires_grad=True)
        # self.xavier_uniform_init(self.learnable_vectors) 

        # GPD head hyperparameters
        self.gpd_xi_max = getattr(args, "gpd_xi_max", 0.5)
        self.gpd_sigma_min = getattr(args, "gpd_sigma_min", 1.0e-3)
        self.gpd_sigma_max = getattr(args, "gpd_sigma_max", 1.0e3)
        self.gpd_eps = getattr(args, "gpd_eps", 1.0e-6)
        self.gpd_nll_weight = getattr(args, "gpd_nll_weight", 1.0)
        self.gpd_debug = getattr(args, "gpd_debug", False)

        

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
    
    
    def forward(self, view1, graph):
        # print(f"view1.shape: {view1.dtype}, {view1.device}")  

        if self.dataset == "NYCBike1":  ## view1.shape: torch.Size([32, 9, 200, 2])
            # view1B = view1[:, :5, :, :]
            # view1A = view1[:, 5:9, :, :]
            view1A = view1[:, -4:19, :, :]
            view1B = view1[:, -9:-4, :, :]
            # view1 = view1[:, -4:19, :, :]
        elif self.dataset == "NYCBike2" or self.dataset == "NYCTaxi" or self.dataset == "BJTaxi":   ## view1.shape: torch.Size([32, 17, 200, 2])
            ## when using input length = 35, these are C and D
            # view1B = view1[:, :9, :, :]
            # view1A = view1[:, 9:17, :, :]
            ## these are A and B
            view1A = view1[:, -8:35, :, :]
            view1B = view1[:, -17:-8, :, :]
            
            
            # view1 = view1[:, -8:35, :, :]
        view1A = view1A.to(self.args.device)
        view1B = view1B.to(self.args.device)
        # print(f"view1.shape: {view1.shape}, view1A.shape: {view1A.shape}, view1B.shape: {view1B.shape}")  ## view1.shape: torch.Size([32, 17, 200, 2]), view1A.shape: torch.Size([32, 8, 200, 2]), view1B.shape: torch.Size([32, 9, 200, 2])
        
        B, T, N, D = view1.size()

        learnable_graph = self.neighbours   ## make 1st channel dimension for einsum to properly message pass
            
        """ check einsum implementation for message passing, is running but probly wrong """
        repr1A = self.encoderA(view1A, learnable_graph) # view1: n,l,v,c; graph: v,v 
        repr1B = self.encoderB(view1B, learnable_graph) # view1: n,l,v,c; graph: v,v 
        
        # print(f"repr1A.shape: {repr1A.shape}, repr1B.shape: {repr1B.shape}")
        
        combined_repr = torch.cat((repr1A, repr1B), dim=3)            ## combine along the channel dimension d_model
        
        
        if self.self_attention_flag:
            combined_repr = self.attentive_fuse(combined_repr)
        
        # o_tilde = self.predict_o_tilde(combined_repr)

        # view1A = torch.cat((view1A, o_tilde), dim=1)
        # repr1A_cls = self.encoderA_cls(view1A, learnable_graph) # view1: n,l,v,c; graph: v,v 
        # repr1B_cls = self.encoderB_cls(view1B, learnable_graph) # view1: n,l,v,c; graph: v,v 
        # combined_repr_cls = torch.cat((repr1A_cls, repr1B_cls), dim=3)            ## combine along the channel dimension d_model
        # combined_repr_cls = self.attentive_fuse_cls(combined_repr_cls)
        # # print(f"combined_repr_cls.shape: {combined_repr_cls.shape}, combined_repr.shape: {combined_repr.shape}")
        repr2 = None
        combined_repr_cls = None
        return combined_repr, combined_repr_cls


    def fetch_spatial_sim(self):
        """
        Fetch the region similarity matrix generated by region embedding.
        Note this can be called only when spatial_sim is True.
        :return sim_mx: tensor, similarity matrix, (v, v)
        """
        return self.encoder.s_sim_mx.cpu()
    
    def fetch_temporal_sim(self):
        return self.encoder.t_sim_mx.cpu()

    def get_gpd_params(self, z1):
        """
        get GPD params (xi, sigma) for each node and timestep
        """
        ## z1.shape: torch.Size([32, 1, 200, 128])
        k = self.ff_key_projection_bias(z1)
        params = torch.matmul(k, self.learnable_vectors_bias)
        xi_raw = params[..., 0:2]
        sigma_raw = params[..., 2:4]
        xi = self.gpd_xi_max * torch.tanh(xi_raw)
        sigma = F.softplus(sigma_raw) + self.gpd_sigma_min
        if self.gpd_sigma_max is not None:
            sigma = torch.clamp(sigma, max=self.gpd_sigma_max)
        return xi, sigma

    def get_bias(self, z1):
        """
        Backward-compatible wrapper: returns concatenated (xi, sigma).
        """
        xi, sigma = self.get_gpd_params(z1)
        return torch.cat([xi, sigma], dim=-1)

    def classify_evs(self, z1, z1_cls):
        """
        classify each next prediction as EV or not
        use separate backbone, and prediction as input. 
        """
        evs = self.get_evs(z1)
        # threshold evs at 0.5
        # evs = (evs > 0.5).float()
        return evs


    def get_evs(self, z1):
        """
        classify each next prediction as EV or not
        """
        return torch.sigmoid(self.mlp_cls(self.ff_to_cls(z1)))

    def predict(self, z1, z1_cls, phase, threshold=None, t=None):
        '''Predicting future traffic flow.
        :param z1, z2 (tensor): shape nvc
        :return: nlvc, l=1, c=2
        '''
        # print("z1.shape: ", z1.shape)
        o_tilde = self.mlp(z1)
        xi, sigma = self.get_gpd_params(z1)
        # o_tilde = scaler.inverse_transform(o_tilde)
        # bias = scaler.inverse_transform(bias)
        # evs = self.classify_evs(z1, z1_cls).detach()
        evs = self.classify_evs(z1, z1_cls)
        if t is not None:
            evs = (evs > t).float()
        if threshold is None:
            threshold = 0.0
        mean_excess = sigma / torch.clamp(1 - xi, min=self.gpd_eps)
        gpd_pred = threshold + mean_excess
        ## which repr to use to calculate the bias, maybe both
        if phase == "pred":
            return o_tilde
        elif phase == "cls":
            return o_tilde
        elif phase == "bias" or phase == "pred_2":
            return o_tilde + evs * (gpd_pred - o_tilde)
        else:
            raise ValueError("phase not recognized")
     
    
    def predict_o_tilde(self, z1):
        '''Predicting future traffic flow.
        :param z1, z2 (tensor): shape nvc
        :return: nlvc, l=1, c=2
        '''
        # print("z1.shape: ", z1.shape)
        o_tilde = self.mlp(z1)
        return o_tilde.detach()
    
    # def classification_loss(self, z1, z1_cls, evs_gt):
    #     evs = self.classify_evs(z1, z1_cls)
    #     return self.focal_loss(evs, evs_gt)
    
    def classification_loss(self, z1, z1_cls, evs_gt, y_true):
        evs = self.classify_evs(z1, z1_cls)
        return F.binary_cross_entropy(evs, evs_gt)
    
    # def classification_loss(self, z1, z1_cls, evs_gt, y_true):
    #     evs = self.classify_evs(z1, z1_cls)
    #     return self.masked_bce(evs, evs_gt, y_true, mask_value=5.0)
    
    # def masked_bce(self, evs, evs_gt, true, mask_value=None):
    #     if mask_value is not None:
    #         mask = torch.gt(true, mask_value)
    #         evs_masked = torch.masked_select(evs, mask)
    #         evs_gt_masked = torch.masked_select(evs_gt, mask)
    #     # Ensure no empty tensors
    #     if evs.numel() == 0 or evs_gt.numel() == 0:
    #         print("\nWarning: Empty tensor after masking")
    #         return F.binary_cross_entropy(evs, evs_gt)

    #     # Check for NaNs in input tensors
    #     if torch.isnan(evs).any() or torch.isnan(evs_gt).any():
    #         print("\nWarning: NaNs in input tensors")
    #         return F.binary_cross_entropy(evs, evs_gt)
            
    #     return F.binary_cross_entropy(evs_masked, evs_gt_masked)
    
    def pred_loss(self, z1, z1_cls, evs_gt, y_true, threshold, scaler, phase, val=False):
        preds = self.predict(z1, z1_cls, phase, threshold=threshold)
        y_pred = scaler.inverse_transform(preds)
        y_true = scaler.inverse_transform(y_true)

        if val:
            pred_loss = self.args.yita * self.loss_fun_val(y_pred[..., 0], y_true[..., 0]) + \
                    (1 - self.args.yita) * self.loss_fun_val(y_pred[..., 1], y_true[..., 1])
        else:
            pred_loss = self.args.yita * self.loss_fun(y_pred[..., 0], y_true[..., 0]) + \
                    (1 - self.args.yita) * self.loss_fun(y_pred[..., 1], y_true[..., 1])

        loss = pred_loss
        return loss
    
    def gpd_nll(self, exceedance, xi, sigma):
        # Negative log-likelihood for GPD
        xi_abs = torch.abs(xi)
        # avoid division by zero in the general formula
        xi_safe = torch.where(xi_abs < 1.0e-3, torch.full_like(xi, 1.0e-3), xi)
        t = 1 + xi * exceedance / sigma
        t = torch.clamp(t, min=self.gpd_eps)
        nll_general = torch.log(sigma) + (1 / xi_safe + 1) * torch.log(t)
        nll_exp = torch.log(sigma) + exceedance / sigma  # xi -> 0
        return torch.where(xi_abs < 1.0e-6, nll_exp, nll_general)
    

    def loss(self, z1, z1_cls, evs, y_true, threshold, scaler, loss_weights, phase, val=False, debug_tag=None):
        l_pred = self.pred_loss(z1, z1_cls, evs, y_true, threshold, scaler, phase, val=val)
        
        l_class = self.classification_loss(z1, z1_cls, evs, y_true)
        # total_loss = l_pred + l_class
        # pred_weight = l_class / total_loss
        # cls_weight = l_pred / total_loss

        # Normalize weights to keep the sum constant, e.g., sum to 2
        # weight_sum = pred_weight + cls_weight
        # pred_weight = 2 * (pred_weight / weight_sum)
        # cls_weight = 2 * (cls_weight / weight_sum)

        # loss_weights = [pred_weight.item(), cls_weight.item()]
        # loss_weights = [1.0, 1.0]
        if phase == "pred":
            loss = loss_weights[0]*l_pred
        else:
            loss = loss_weights[0]*l_pred + loss_weights[1]*l_class
        # loss = loss_weights[0]*l_pred

        if phase == "bias" or phase == "pred_2":
            xi, sigma = self.get_gpd_params(z1)
            exceedance = torch.clamp(y_true - threshold, min=0.0)
            mask = (evs > 0.5) & (exceedance > 0)
        
            if mask.any().item():
                gpd_nll = self.gpd_nll(exceedance[mask], xi[mask], sigma[mask]).mean()
                loss = loss + self.gpd_nll_weight * gpd_nll

        l_pred=l_pred.item()
        l_class=l_class.item()
        return loss, l_pred, l_class, loss_weights
    
    """
    # def classification_loss(self, z1, evs_gt):
    #     evs = self.get_evs(z1)
        
    #     # Calculate the total number of elements and number of positives (extremes)
    #     total_elements = evs_gt.numel()
    #     num_extremes = evs_gt.sum()
    #     num_non_extremes = total_elements - num_extremes

    #     # Compute weights for each class
    #     weight_for_1 = total_elements / (num_extremes + 1e-6)  # Adding a small constant to avoid division by zero
    #     weight_for_0 = total_elements / (num_non_extremes + 1e-6)

    #     # Create a tensor of weights that matches the shape of evs_gt
    #     weights = evs_gt.float() * weight_for_1 + (1 - evs_gt.float()) * weight_for_0

    #     # Calculate the weighted binary cross entropy loss
    #     return F.binary_cross_entropy(evs, evs_gt, weight=weights)
    """
    
    def focal_loss(self, inputs, targets):
        """ Compute the focal loss given inputs and targets:
        
        inputs: tensor of predictions (probability of being the positive class)
        targets: tensor of target labels {0, 1}
        """
        # First, compute the binary cross-entropy loss without reduction
        alpha, gamma = 0.25, 2.0
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')

        # Here we calculate p_t
        p_t = targets * inputs + (1 - targets) * (1 - inputs)

        # Calculate the factor (1 - p_t)^gamma
        loss_factor = (1 - p_t) ** gamma

        # Calculate final focal loss
        focal_loss = alpha * loss_factor * bce_loss

        return focal_loss.mean()

    # def classification_loss(self, z1, evs_gt):
    #     z1_detached = z1.detach()
    #     evs = self.get_evs(z1_detached)
    #     return self.focal_loss(evs, evs_gt)
    
    
    def temporal_loss(self, z1, z2):
        return self.thm(z1, z2)

    def spatial_loss(self, z1, z2):
        return self.shm(z1, z2)
    
