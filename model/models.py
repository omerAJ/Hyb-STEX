import torch.nn as nn
import torch
# import 
from lib.utils import masked_mae_loss, masked_mse_loss
from model.aug import (
    aug_topology, 
    aug_traffic, 
)

from model.layers import (
    STEncoder, 
    MLP,
    self_Attention,
    PositionwiseFeedForward,
)

from model.vision_transformer_utils import apply_masks_targets
import torch.nn.functional as F
import numpy as np
class STSSL(nn.Module):
    def __init__(self, args):
        super(STSSL, self).__init__()
        self.args = args

        self.attention1 = self_Attention(int((2)*args.d_model), 4)
        self.attention2 = self_Attention(int((2)*args.d_model), 4)
        
        self.ff = PositionwiseFeedForward(d_model=128, d_ff=64*4)
        self.mlp = MLP(int((2)*args.d_model), args.d_output)
        self.mlp_classifier = MLP(int((2)*args.d_model), args.d_output)
        if args.loss == 'mae':
            self.loss_fun = masked_mae_loss(mask_value=5.0)
        elif args.loss == 'mse':
            self.loss_fun = masked_mse_loss(mask_value=5.0)
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
        
        # ## norms
        self.layernorm1 = nn.LayerNorm(int((2)*args.d_model))
        self.layernorm2 = nn.LayerNorm(int((2)*args.d_model))
        self.layernorm3 = nn.LayerNorm(int((2)*args.d_model))
        

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

        T = 8
        N = 200
        self.weights = nn.Parameter(torch.ones(N) / N)
        self.key_projection = nn.Linear(int((2)*args.d_model), int((2)*args.d_model))
        # self.project_to_classify = nn.Linear(int((2)*args.d_model), int((2)*args.d_model))
        self.learnable_vectors = nn.Parameter(torch.zeros(1, 1, 128, 2), requires_grad=True)
        self.ff_to_classify = PositionwiseFeedForward(d_model=int((2)*args.d_model), d_ff=int((2)*args.d_model)*4)

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
        
        if self.dataset == "NYCBike1":
            view1A = view1[:, -4:19, :, :]
            view1B = view1[:, -9:-4, :, :]
            # view1 = view1[:, -4:19, :, :]
        elif self.dataset == "NYCBike2" or self.dataset == "NYCTaxi" or self.dataset == "BJTaxi": 
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
        # learnable_graph = None
        return combined_repr

    def fetch_spatial_sim(self):
        """
        Fetch the region similarity matrix generated by region embedding.
        Note this can be called only when spatial_sim is True.
        :return sim_mx: tensor, similarity matrix, (v, v)
        """
        return self.encoder.s_sim_mx.cpu()
    
    def fetch_temporal_sim(self):
        return self.encoder.t_sim_mx.cpu()

    def get_bias(self, z1):
        """
        get the bias for each node and timestep 
        """
        k = self.key_projection(z1)
        bias = torch.matmul(k, self.learnable_vectors)
        return bias

    def get_evs(self, z1):
        """
        classify each next prediction as EV or not
        """
        return torch.sigmoid(self.mlp_classifier(self.ff_to_classify(z1)))

    def predict(self, z1):
        '''Predicting future traffic flow.
        :param z1, z2 (tensor): shape nvc
        :return: nlvc, l=1, c=2
        '''
        # print("z1.shape: ", z1.shape)
        o_tilde = self.mlp(z1)
        evs = self.get_evs(z1)
        o = o_tilde + self.get_bias(z1) * evs
        return o

    def loss(self, z1, evs, y_true, scaler, loss_weights):
        l_pred = self.pred_loss(z1, evs, y_true, scaler)
        
        l_class = self.classification_loss(z1, evs)
        # sep_loss = [l1.item()]
        loss = l_class + l_pred

        l_pred=l_pred.item()
        l_class=l_class.item()
        return loss, l_pred, l_class

    def mae_torch(self, pred, true, mask_value=None):
        if mask_value != None:
            # print(f"true.device: {true.device}, pred.device: {pred.device}")
            
            mask = torch.gt(true, mask_value)
            # nodesMasked=mask[mask==True].shape[0]
            # print("total nodes masked", nodesMasked, "/4096",  "nodes on average masked in each sample: ", nodesMasked/true.shape[0])
            # print(f"pred.shape: {pred.shape}")
            diff = torch.abs(true-pred)
            # weights = torch.sigmoid(self.weights)
            weights = None
            if self.args.loss == 'mae':
                w = torch.abs(true-pred)
                diff = diff * w
            diff = torch.masked_select(diff, mask)
            
            # pred = torch.masked_select(pred, mask)
            # print(f"pred.shape: {pred.shape}")
            
            # true = torch.masked_select(true, mask)

        # l = torch.abs(true-pred)
        
        # print(f"l.shape: {l.shape}, self.weights.shape: {self.weights.shape}")
        return torch.mean(diff)
    

    def masked_mae_loss(self, mask_value):
        def loss(preds, labels):
            mae = self.mae_torch(pred=preds, true=labels, mask_value=mask_value)
            return mae
        return loss
    
    # def classification_loss(self, z1, evs_gt):
    #     evs = self.get_evs(z1)
    #     return F.binary_cross_entropy(evs, evs_gt)
    
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

    def classification_loss(self, z1, evs_gt):
        evs = self.get_evs(z1)
        return self.focal_loss(evs, evs_gt)
    
    # def classification_loss(self, z1, evs_gt):
    #     z1_detached = z1.detach()
    #     evs = self.get_evs(z1_detached)
    #     return self.focal_loss(evs, evs_gt)
    
    def pred_loss(self, z1, evs_gt, y_true, scaler):
        preds = self.predict(z1)
        y_pred = scaler.inverse_transform(preds)
        y_true = scaler.inverse_transform(y_true)

        pred_loss = self.args.yita * self.loss_fun(y_pred[..., 0], y_true[..., 0]) + \
                (1 - self.args.yita) * self.loss_fun(y_pred[..., 1], y_true[..., 1])

        
        loss = pred_loss
        return loss
    
    def temporal_loss(self, z1, z2):
        return self.thm(z1, z2)

    def spatial_loss(self, z1, z2):
        return self.shm(z1, z2)
    