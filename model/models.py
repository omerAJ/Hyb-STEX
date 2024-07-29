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
    attentive_fusion,
    classify_evs,
)

from model.vision_transformer_utils import apply_masks_targets
import torch.nn.functional as F
import numpy as np
class STSSL(nn.Module):
    def __init__(self, args):
        super(STSSL, self).__init__()
        self.args = args

        # self.attention1 = self_Attention(int((2)*args.d_model), 4)
        # self.attention2 = self_Attention(int((2)*args.d_model), 4)
        
        self.attentive_fuse = attentive_fusion(args.d_model)
        self.attentive_fuse_cls = attentive_fusion(args.d_model)

        self.ff = PositionwiseFeedForward(d_model=128, d_ff=64*4)
        self.mlp = MLP(int((2)*args.d_model), args.d_output)
        self.mlp_classifier = MLP(int((2)*args.d_model), args.d_output)
        self.mlp_bias = MLP(int((2)*args.d_model), args.d_output)
        if args.loss == 'mae':
            self.loss_fun = masked_mae_loss(mask_value=5.0)
            self.loss_fun_cls = masked_mae_loss(mask_value=5.0)
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
    
        self.encoderA_cls = classify_evs(Kt=3, Ks=args.cheb_order, blocks=[[2, int(args.d_model//2), args.d_model], [args.d_model, int(args.d_model//2), args.d_model]], 
                        input_length=args.input_length, num_nodes=args.num_nodes, droprate=args.dropout, graph_init=graph_init)
        self.encoderB_cls = classify_evs(Kt=3, Ks=args.cheb_order, blocks=[[2, int(args.d_model//2), args.d_model], [args.d_model, int(args.d_model//2), args.d_model]], 
                        input_length=args.input_length, num_nodes=args.num_nodes, droprate=args.dropout, graph_init=graph_init)         
        
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
        self.project_to_classify = nn.Linear(int((2)*args.d_model), int((2)*args.d_model))
        self.learnable_vectors = nn.Parameter(torch.zeros(1, 1, 128, 2), requires_grad=True)
        # self.xavier_uniform_init(self.learnable_vectors) 

    def xavier_uniform_init(self, tensor):
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
        std = np.sqrt(2.0 / (fan_in + fan_out))
        nn.init.uniform_(tensor, -std, std) 
    
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
        
        repr1A = self.encoderA(view1A, graph) # view1: n,l,v,c; graph: v,v 
        repr1B = self.encoderB(view1B, graph) # view1: n,l,v,c; graph: v,v 
        
        repr1A_cls = self.encoderA_cls(view1A, graph) # view1: n,l,v,c; graph: v,v 
        repr1B_cls = self.encoderB_cls(view1B, graph) # view1: n,l,v,c; graph: v,v 
        # print(f"repr1A.shape: {repr1A.shape}, repr1B.shape: {repr1B.shape}")
        
        combined_repr = torch.cat((repr1A, repr1B), dim=3)            ## combine along the channel dimension d_model
        combined_repr_cls = torch.cat((repr1A_cls, repr1B_cls), dim=3)            ## combine along the channel dimension d_model
        
        if self.self_attention_flag:
            combined_repr = self.attentive_fuse(combined_repr)
            combined_repr_cls = self.attentive_fuse_cls(combined_repr_cls)
        
        return combined_repr, combined_repr_cls


    def get_bias(self, z1):
        """
        get the bias for each node and timestep 
        """
        # k = self.key_projection(z1)
        # bias = torch.matmul(k, self.learnable_vectors)
        bias = self.mlp_bias(z1)
        return bias

    
    def predict(self, z1, z1_cls):
        '''Predicting future traffic flow.
        :param z1, z2 (tensor): shape nvc
        :return: nlvc, l=1, c=2
        '''
        # print("z1.shape: ", z1.shape)
        o_tilde = self.mlp(z1)
        bias = self.get_bias(z1_cls)
        
        return o_tilde, bias

    
    def predict_final(self, z1, z1_cls):
        '''Predicting future traffic flow.
        :param z1, z2 (tensor): shape nvc
        :return: nlvc, l=1, c=2
        '''
        # print("z1.shape: ", z1.shape)
        o_tilde, bias = self.predict(z1, z1_cls)
        return o_tilde+bias

    def loss(self, z1, z1_cls, evs, y_true, scaler, loss_weights):
        l_pred, l_bias = self.pred_loss(z1, z1_cls, evs, y_true, scaler)
        # print(f"l_pred: {l_pred}, l_bias: {l_bias}")
        # print(f"l_pred: {l_pred}, l_bias: {l_bias}")
        # print(f"loss_weights: {loss_weights}")
        loss = loss_weights[0]*l_pred + loss_weights[1]*l_bias
        # loss = 0*l_pred + 0*l_bias

        l_pred=l_pred.item()
        l_bias=l_bias.item()
        
        return loss, l_pred, l_bias
    
    def loss_val(self, z1, z1_cls, evs, y_true, scaler, loss_weights):
        l_pred, l_bias, l_total = self.pred_loss_val(z1, z1_cls, evs, y_true, scaler)
        
        l_total = l_total
        l_pred=l_pred.item()
        l_bias=l_bias.item()
        
        return l_total, l_pred, l_bias
    
    
    def pred_loss_val(self, z1, z1_cls, evs_gt, y_true, scaler):
        o_tilde, bias = self.predict(z1, z1_cls)
        
        residual = y_true - o_tilde
        ## y_true = o_tilde + bias, -> bias must track y_true-o_tilde
        o_tilde = scaler.inverse_transform(o_tilde)
        y_true = scaler.inverse_transform(y_true)
        residual = scaler.inverse_transform(residual)

        ## loss for the main head
        pred_loss = self.args.yita * self.loss_fun(o_tilde[..., 0], y_true[..., 0]) + \
                (1 - self.args.yita) * self.loss_fun(o_tilde[..., 1], y_true[..., 1])

        ## loss for the bias head
        bias_loss = self.args.yita * self.loss_fun(bias[..., 0], residual[..., 0]) + \
                (1 - self.args.yita) * self.loss_fun(bias[..., 1], residual[..., 1])
        
        ## loss for the total
        prediction = o_tilde + bias
        total_loss = self.args.yita * self.loss_fun(prediction[..., 0], y_true[..., 0]) + \
                (1 - self.args.yita) * self.loss_fun(prediction[..., 1], y_true[..., 1])
        
        return pred_loss, bias_loss, total_loss
    
    def pred_loss(self, z1, z1_cls, evs_gt, y_true, scaler):
        o_tilde, bias = self.predict(z1, z1_cls)
        o_tilde_detached = o_tilde.detach()
        residual = y_true - o_tilde_detached
        residual = residual.detach()
        # check if there is any nan value in residual
        if torch.isnan(residual).any():
            print("residual has nan values")

        ## y_true = o_tilde + bias, -> bias must track y_true-o_tilde
        o_tilde = scaler.inverse_transform(o_tilde)
        y_true = scaler.inverse_transform(y_true)
        residual = scaler.inverse_transform(residual)

        ## loss for the main head
        pred_loss = self.args.yita * self.loss_fun(o_tilde[..., 0], y_true[..., 0]) + \
                (1 - self.args.yita) * self.loss_fun(o_tilde[..., 1], y_true[..., 1])

        ## loss for the bias head
        bias_loss = self.args.yita * self.loss_fun_cls(bias[..., 0], residual[..., 0]) + \
                (1 - self.args.yita) * self.loss_fun_cls(bias[..., 1], residual[..., 1])
        
        return pred_loss, bias_loss
    