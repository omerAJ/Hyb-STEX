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

        N = args.num_nodes
        self.weights = nn.Parameter(torch.ones(N) / N)
        self.ff_to_cls = PositionwiseFeedForward(d_model=128, d_ff=128*4)
        self.ff_to_gpd = PositionwiseFeedForward(d_model=128, d_ff=64*4)
        self.learnable_vectors_gpd = nn.Parameter(torch.zeros(1, 1, 128, 4), requires_grad=True)

        self.tail_lambda_cls = getattr(args, "tail_lambda_cls", 1.0)
        self.tail_lambda_gpd = getattr(args, "tail_lambda_gpd", 1.0)
        self.tail_threshold_q = getattr(args, "tail_threshold_q", 0.90)
        self.tail_xi_min = getattr(args, "tail_xi_min", -0.5)
        self.tail_xi_max = getattr(args, "tail_xi_max", -0.02)
        self.tail_eps = getattr(args, "tail_eps", 1.0e-6)
        self.tail_schedule = getattr(args, "tail_schedule", "static")
        self.register_buffer("tail_u", torch.zeros(1, 1, args.num_nodes, args.d_output))

        

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

    def set_tail_thresholds(self, thresholds):
        if thresholds.dim() == 2:
            thresholds = thresholds.unsqueeze(0).unsqueeze(0)
        self.tail_u.copy_(thresholds.to(device=self.tail_u.device, dtype=self.tail_u.dtype))

    def predict_base(self, z1):
        return self.mlp(z1)

    def predict_o_tilde(self, z1):
        return self.predict_base(z1).detach()

    def get_classifier_logits(self, z1):
        return self.mlp_cls(self.ff_to_cls(z1))

    def classify_evs(self, z1, z1_cls=None):
        return torch.sigmoid(self.get_classifier_logits(z1))

    def get_tail_raw_params(self, z1):
        projected = self.ff_to_gpd(z1)
        params = torch.matmul(projected, self.learnable_vectors_gpd)
        raw_sigma = params[..., :self.args.d_output]
        raw_xi = params[..., self.args.d_output:]
        return raw_sigma, raw_xi

    def get_tail_params(self, z1):
        raw_sigma, raw_xi = self.get_tail_raw_params(z1)
        sigma = F.softplus(raw_sigma) + 1.0e-4
        xi = self.tail_xi_min + (self.tail_xi_max - self.tail_xi_min) * torch.sigmoid(raw_xi)
        return raw_sigma, raw_xi, sigma, xi

    def get_tail_prediction_components(self, z1, scaler):
        y_hat = self.predict_base(z1)
        y_hat_orig = scaler.inverse_transform(y_hat)
        logit_q = self.get_classifier_logits(z1)
        q = torch.sigmoid(logit_q)
        raw_sigma, raw_xi, sigma, xi = self.get_tail_params(z1)
        delta = q * (self.tail_u + sigma / torch.clamp(1 - xi, min=self.tail_eps))
        y_corr = torch.clamp_min(y_hat_orig + delta, 0.0)
        return {
            "y_hat": y_hat,
            "y_hat_orig": y_hat_orig,
            "logit_q": logit_q,
            "q": q,
            "raw_sigma": raw_sigma,
            "raw_xi": raw_xi,
            "sigma": sigma,
            "xi": xi,
            "delta": delta,
            "y_corr": y_corr,
        }

    def build_tail_targets(self, y_hat_orig, y_true_orig):
        residual = y_true_orig - y_hat_orig
        positive_residual = torch.clamp(residual, min=0.0)
        indicator = (positive_residual > self.tail_u).float()
        exceedance = torch.clamp(positive_residual - self.tail_u, min=0.0)
        exceedance = exceedance * indicator
        return indicator, exceedance, positive_residual

    def predict(self, z1, z1_cls, phase, scaler=None, t=None):
        if phase == "pred":
            return self.predict_base(z1)
        if phase == "tail":
            if scaler is None:
                raise ValueError("scaler is required for tail prediction.")
            components = self.get_tail_prediction_components(z1, scaler)
            if t is not None:
                gate = (components["q"] > t).float()
                delta = gate * (self.tail_u + components["sigma"] / torch.clamp(1 - components["xi"], min=self.tail_eps))
                return torch.clamp_min(components["y_hat_orig"] + delta, 0.0)
            return components["y_corr"]
        raise ValueError("phase not recognized")

    def weighted_reconstruction_loss(self, y_pred, y_true, val=False):
        loss_fn = self.loss_fun_val if val else self.loss_fun
        return self.args.yita * loss_fn(y_pred[..., 0], y_true[..., 0]) + \
            (1 - self.args.yita) * loss_fn(y_pred[..., 1], y_true[..., 1])

    @staticmethod
    def _interpolate_segment(progress, start_progress, end_progress, start_value, end_value):
        if end_progress <= start_progress:
            return end_value
        scaled_progress = (progress - start_progress) / (end_progress - start_progress)
        scaled_progress = min(max(scaled_progress, 0.0), 1.0)
        return start_value + (end_value - start_value) * scaled_progress

    def get_tail_loss_weights(self, epoch=None, total_epochs=None):
        base_lambda_cls = float(self.tail_lambda_cls)
        base_lambda_gpd = float(self.tail_lambda_gpd)
        schedule = self.tail_schedule

        if schedule == "static" or epoch is None or total_epochs is None:
            return {
                "schedule": schedule,
                "lambda_cls": base_lambda_cls,
                "lambda_gpd": base_lambda_gpd,
            }

        if total_epochs <= 1:
            progress = 1.0
        else:
            progress = (epoch - 1) / float(total_epochs - 1)

        schedule_boundaries = {
            "soft_ramp_fast": (0.15, 0.60, 1.0, 0.3, 0.3, 0.0),
            "soft_ramp_balanced": (0.25, 0.70, 1.0, 0.5, 0.5, 0.0),
            "soft_ramp_long": (0.35, 0.80, 1.0, 0.2, 0.2, 0.0),
        }
        if schedule not in schedule_boundaries:
            raise ValueError(f"Unsupported tail schedule: {schedule}")

        warmup_end, mixed_end, final_end, mixed_cls_end, final_cls_start, final_cls_end = schedule_boundaries[schedule]
        if progress < warmup_end:
            cls_multiplier = 1.0
            gpd_multiplier = 0.0
        elif progress < mixed_end:
            cls_multiplier = self._interpolate_segment(
                progress,
                warmup_end,
                mixed_end,
                1.0,
                mixed_cls_end,
            )
            gpd_multiplier = self._interpolate_segment(
                progress,
                warmup_end,
                mixed_end,
                0.0,
                1.0,
            )
        else:
            cls_multiplier = self._interpolate_segment(
                progress,
                mixed_end,
                final_end,
                final_cls_start,
                final_cls_end,
            )
            gpd_multiplier = 1.0

        return {
            "schedule": schedule,
            "lambda_cls": base_lambda_cls * cls_multiplier,
            "lambda_gpd": base_lambda_gpd * gpd_multiplier,
        }

    def classification_loss(self, logit_q, indicator):
        return F.binary_cross_entropy_with_logits(logit_q, indicator)

    def gpd_loss(self, exceedance, indicator, xi, sigma):
        exceedance_mask = indicator > 0.5
        exceedance_count = int(exceedance_mask.sum().item())
        zero = sigma.new_zeros(())
        if exceedance_count == 0:
            return zero, {
                "exceedance_count": 0,
                "valid_exceedance_count": 0,
                "invalid_support_count": 0,
            }

        safe_sigma = torch.clamp(sigma, min=self.tail_eps)
        term = 1 + xi * exceedance / safe_sigma
        valid_mask = exceedance_mask & (term > self.tail_eps)
        valid_exceedance_count = int(valid_mask.sum().item())
        invalid_support_count = exceedance_count - valid_exceedance_count
        if valid_exceedance_count == 0:
            return zero, {
                "exceedance_count": exceedance_count,
                "valid_exceedance_count": 0,
                "invalid_support_count": invalid_support_count,
            }

        loss = torch.log(safe_sigma[valid_mask]) + (1 / xi[valid_mask] + 1) * torch.log(torch.clamp(term[valid_mask], min=self.tail_eps))
        return loss.mean(), {
            "exceedance_count": exceedance_count,
            "valid_exceedance_count": valid_exceedance_count,
            "invalid_support_count": invalid_support_count,
        }

    def loss(self, z1, z1_cls, evs, y_true, scaler, loss_weights, phase, val=False):
        del evs
        y_true_orig = scaler.inverse_transform(y_true)

        if phase == "pred":
            y_hat = self.predict_base(z1)
            y_hat_orig = scaler.inverse_transform(y_hat)
            pred_mae = self.weighted_reconstruction_loss(y_hat_orig, y_true_orig, val=val)
            metrics = {
                "pred_mae": pred_mae.item(),
                "cls_loss": 0.0,
                "gpd_loss": 0.0,
                "selection_mae": pred_mae.item(),
                "exceedance_count": 0,
                "valid_exceedance_count": 0,
                "invalid_support_count": 0,
            }
            return pred_mae, metrics

        if phase != "tail":
            raise ValueError("phase not recognized")

        tail_loss_weights = loss_weights or self.get_tail_loss_weights()
        components = self.get_tail_prediction_components(z1, scaler)
        indicator, exceedance, _ = self.build_tail_targets(components["y_hat_orig"].detach(), y_true_orig)
        cls_loss = self.classification_loss(components["logit_q"], indicator)
        gpd_loss, gpd_stats = self.gpd_loss(exceedance, indicator, components["xi"], components["sigma"])
        tail_loss = tail_loss_weights["lambda_cls"] * cls_loss + tail_loss_weights["lambda_gpd"] * gpd_loss
        corrected_mae = self.weighted_reconstruction_loss(components["y_corr"], y_true_orig, val=val)
        metrics = {
            "pred_mae": corrected_mae.item(),
            "cls_loss": cls_loss.item(),
            "gpd_loss": gpd_loss.item(),
            "selection_mae": corrected_mae.item(),
            "exceedance_count": gpd_stats["exceedance_count"],
            "valid_exceedance_count": gpd_stats["valid_exceedance_count"],
            "invalid_support_count": gpd_stats["invalid_support_count"],
            "lambda_cls": float(tail_loss_weights["lambda_cls"]),
            "lambda_gpd": float(tail_loss_weights["lambda_gpd"]),
            "tail_schedule": tail_loss_weights["schedule"],
        }
        return tail_loss, metrics
    
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
    
