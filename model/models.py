import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.utils import masked_frechet_loss, masked_gumbell_loss, masked_mae_loss, masked_mse_loss
from model.layers import SpatioConvLayer


class PEMS04TemporalConv(nn.Module):
    """A same-padding gated temporal convolution with residual projection."""

    def __init__(self, c_in, c_out, kernel_size=3):
        super().__init__()
        self.c_out = c_out
        self.residual_proj = nn.Conv2d(c_in, c_out, kernel_size=1) if c_in != c_out else None
        self.conv = nn.Conv2d(
            c_in,
            c_out * 2,
            kernel_size=(kernel_size, 1),
            padding=(kernel_size // 2, 0),
        )

    def forward(self, x):
        residual = self.residual_proj(x) if self.residual_proj is not None else x
        gated = self.conv(x)
        return (gated[:, : self.c_out] + residual) * torch.sigmoid(gated[:, self.c_out :])


class PEMS04STBlock(nn.Module):
    """Two temporal convolutions around a Chebyshev spatial conv, without collapsing time."""

    def __init__(self, d_model, cheb_order, dropout):
        super().__init__()
        self.temporal1 = PEMS04TemporalConv(d_model, d_model)
        self.spatial = SpatioConvLayer(cheb_order, d_model, d_model)
        self.spatial_mix = nn.Parameter(torch.tensor(1.0))
        self.temporal2 = PEMS04TemporalConv(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cheb_laplacian):
        h = x.permute(0, 3, 1, 2)
        h = self.temporal1(h)

        spatial_residual = h
        h = self.spatial(h, cheb_laplacian)
        mix = torch.sigmoid(self.spatial_mix)
        h = mix * h + (1.0 - mix) * spatial_residual
        x = self.dropout(self.norm1(h.permute(0, 2, 3, 1)))

        h = self.temporal2(x.permute(0, 3, 1, 2))
        return self.dropout(self.norm2(h.permute(0, 2, 3, 1)))


class PEMS04Encoder(nn.Module):
    """A PEMS04-specific spatio-temporal encoder that preserves the full 12-step history."""

    def __init__(self, input_dim, d_model, input_length, num_nodes, cheb_order, dropout):
        super().__init__()
        del input_length
        self.cheb_order = cheb_order
        self.num_nodes = num_nodes
        self.input_proj = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                PEMS04STBlock(d_model=d_model, cheb_order=cheb_order, dropout=dropout),
                PEMS04STBlock(d_model=d_model, cheb_order=cheb_order, dropout=dropout),
            ]
        )
        self.output_norm = nn.LayerNorm(d_model)

    @staticmethod
    def _cal_laplacian(graph):
        identity = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype)
        graph = graph + identity
        degree = torch.diag(torch.sum(graph, dim=-1) ** (-0.5))
        return identity - torch.mm(torch.mm(degree, graph), degree)

    @staticmethod
    def _cheb_polynomial(laplacian, order):
        num_nodes = laplacian.size(0)
        cheb = torch.zeros([order, num_nodes, num_nodes], device=laplacian.device, dtype=laplacian.dtype)
        cheb[0] = torch.eye(num_nodes, device=laplacian.device, dtype=laplacian.dtype)
        if order == 1:
            return cheb
        cheb[1] = laplacian
        for idx in range(2, order):
            cheb[idx] = 2 * torch.mm(laplacian, cheb[idx - 1]) - cheb[idx - 2]
        return cheb

    def forward(self, x, graph):
        if graph.size(0) != self.num_nodes:
            raise ValueError(f"Graph node count mismatch: expected {self.num_nodes}, got {graph.size(0)}")
        cheb_laplacian = self._cheb_polynomial(self._cal_laplacian(graph), self.cheb_order)

        h = self.input_dropout(self.input_norm(self.input_proj(x)))
        for block in self.blocks:
            h = block(h, cheb_laplacian)
        return self.output_norm(h)


class PEMS04HorizonProjection(nn.Module):
    """Project the full temporal latent sequence for each node directly to the forecast horizon."""

    def __init__(self, input_length, d_model, output_length, d_output, dropout, hidden_scale=1.0):
        super().__init__()
        input_dim = input_length * d_model
        hidden_dim = max(int(input_dim * hidden_scale), output_length * d_output)
        self.output_length = output_length
        self.d_output = d_output
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_length * d_output),
        )

    def forward(self, z):
        batch_size, _, num_nodes, _ = z.shape
        flattened = z.transpose(1, 2).reshape(batch_size, num_nodes, -1)
        out = self.proj(flattened)
        out = out.view(batch_size, num_nodes, self.output_length, self.d_output)
        return out.transpose(1, 2)


class STSSL(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dataset = args.dataset
        self.output_length = int(getattr(args, "output_length", 1))

        self.loss_fun_val = masked_mae_loss(mask_value=5.0)
        if args.loss == "mae":
            self.loss_fun = masked_mae_loss(mask_value=5.0)
        elif args.loss == "mse":
            self.loss_fun = masked_mse_loss(mask_value=5.0)
        elif args.loss == "gumbell":
            self.loss_fun = masked_gumbell_loss(mask_value=5.0)
        elif args.loss == "frechet":
            self.loss_fun = masked_frechet_loss(mask_value=5.0)
        else:
            raise ValueError(f"Unsupported loss type: {args.loss}")

        if self.dataset != "PEMS04":
            raise ValueError("This branch is specialized for PEMS04 only.")

        adjacency = np.load(args.graph_file)["adj_mx"].astype(np.float32)
        self.register_buffer("neighbours", torch.from_numpy(adjacency))
        self.encoder = PEMS04Encoder(
            input_dim=args.d_input,
            d_model=args.d_model,
            input_length=args.input_length,
            num_nodes=args.num_nodes,
            cheb_order=args.cheb_order,
            dropout=args.dropout,
        )
        self.mlp = PEMS04HorizonProjection(
            input_length=args.input_length,
            d_model=args.d_model,
            output_length=self.output_length,
            d_output=args.d_output,
            dropout=args.dropout,
            hidden_scale=1.0,
        )
        self.mlp_cls = PEMS04HorizonProjection(
            input_length=args.input_length,
            d_model=args.d_model,
            output_length=self.output_length,
            d_output=args.d_output,
            dropout=args.dropout,
            hidden_scale=0.5,
        )
        self.ff_to_gpd = PEMS04HorizonProjection(
            input_length=args.input_length,
            d_model=args.d_model,
            output_length=self.output_length,
            d_output=args.d_output,
            dropout=args.dropout,
            hidden_scale=0.5,
        )

        self.tail_lambda_cls = getattr(args, "tail_lambda_cls", 1.0)
        self.tail_lambda_gpd = getattr(args, "tail_lambda_gpd", 1.0)
        self.tail_mae_weight = getattr(args, "tail_mae_weight", 0.0)
        self.tail_threshold_q = getattr(args, "tail_threshold_q", 0.90)
        self.tail_xi_min = getattr(args, "tail_xi_min", -0.5)
        self.tail_xi_max = getattr(args, "tail_xi_max", -0.02)
        self.tail_eps = getattr(args, "tail_eps", 1.0e-6)
        self.tail_schedule = getattr(args, "tail_schedule", "static")
        self.tail_magnitude_mode = getattr(args, "tail_magnitude_mode", "point_excess")
        self.tail_classifier_loss_type = getattr(args, "tail_classifier_loss_type", "bce")
        self.tail_pos_weight_multiplier = getattr(args, "tail_pos_weight_multiplier", 1.0)
        self.tail_focal_gamma = getattr(args, "tail_focal_gamma", 2.0)
        self.tail_focal_alpha_pos = getattr(args, "tail_focal_alpha_pos", 0.75)
        self.tail_base_pos_weight = 1.0
        self.tail_effective_pos_weight = 1.0
        self.tail_exceedance_rate = 0.0
        self.tail_positive_count = 0
        self.tail_negative_count = 0
        self.register_buffer(
            "tail_u",
            torch.zeros(1, self.output_length, args.num_nodes, args.d_output),
        )
        self.register_buffer(
            "tail_mean_excess",
            torch.zeros(1, self.output_length, args.num_nodes, args.d_output),
        )
        self._cached_last_flow = None

    def forward(self, view1, graph):
        del graph
        view1 = view1.to(self.args.device)
        self._cached_last_flow = view1[:, -1:, :, [0]]
        repr1 = self.encoder(view1, self.neighbours)
        return repr1, None

    def fetch_spatial_sim(self):
        return None

    def fetch_temporal_sim(self):
        return None

    def set_tail_thresholds(self, thresholds):
        if thresholds.dim() == 2:
            thresholds = thresholds.unsqueeze(0).unsqueeze(0)
        elif thresholds.dim() == 3:
            thresholds = thresholds.unsqueeze(0)
        self.tail_u.copy_(thresholds.to(device=self.tail_u.device, dtype=self.tail_u.dtype))

    def set_tail_mean_excess(self, mean_excess):
        if mean_excess.dim() == 2:
            mean_excess = mean_excess.unsqueeze(0).unsqueeze(0)
        elif mean_excess.dim() == 3:
            mean_excess = mean_excess.unsqueeze(0)
        self.tail_mean_excess.copy_(mean_excess.to(device=self.tail_mean_excess.device, dtype=self.tail_mean_excess.dtype))

    def set_tail_classifier_stats(self, stats):
        self.tail_base_pos_weight = float(stats.get("base_pos_weight", 1.0))
        self.tail_effective_pos_weight = float(stats.get("effective_pos_weight", 1.0))
        self.tail_exceedance_rate = float(stats.get("exceedance_rate", 0.0))
        self.tail_positive_count = int(stats.get("positive_count", 0))
        self.tail_negative_count = int(stats.get("negative_count", 0))

    def predict_base(self, z1):
        if self._cached_last_flow is None:
            raise RuntimeError("PEMS04 persistence skip requires a forward pass before predict_base.")
        learned_residual = self.mlp(z1)
        persistence = self._cached_last_flow.repeat(1, self.output_length, 1, 1)
        return persistence + learned_residual

    def predict_o_tilde(self, z1):
        return self.predict_base(z1).detach()

    def get_classifier_logits(self, z1):
        return self.mlp_cls(z1)

    def classify_evs(self, z1, z1_cls=None):
        del z1_cls
        return torch.sigmoid(self.get_classifier_logits(z1))

    def get_tail_raw_params(self, z1):
        raw_sigma = self.ff_to_gpd(z1)
        raw_xi = torch.zeros_like(raw_sigma)
        return raw_sigma, raw_xi

    def get_tail_params(self, z1):
        raw_sigma, raw_xi = self.get_tail_raw_params(z1)
        sigma = F.softplus(raw_sigma) + 1.0e-4
        xi = self.tail_xi_min + (self.tail_xi_max - self.tail_xi_min) * torch.sigmoid(raw_xi)
        normal_mu = F.softplus(raw_xi)
        return raw_sigma, raw_xi, sigma, xi, normal_mu

    def get_expected_excess(self, sigma, xi, normal_mu):
        if self.tail_magnitude_mode == "gpd":
            return sigma / torch.clamp(1 - xi, min=self.tail_eps)
        if self.tail_magnitude_mode == "normal_excess":
            return normal_mu
        if self.tail_magnitude_mode == "point_excess":
            return sigma
        if self.tail_magnitude_mode == "fixed_mean_excess":
            return self.tail_mean_excess
        if self.tail_magnitude_mode == "threshold_only":
            return torch.zeros_like(self.tail_u)
        raise ValueError(f"Unsupported tail_magnitude_mode: {self.tail_magnitude_mode}")

    def get_tail_prediction_components(self, z1, scaler):
        y_hat = self.predict_base(z1)
        y_hat_orig = scaler.inverse_transform(y_hat)
        logit_q = self.get_classifier_logits(z1)
        q = torch.sigmoid(logit_q)
        raw_sigma, raw_xi, sigma, xi, normal_mu = self.get_tail_params(z1)
        expected_excess = self.get_expected_excess(sigma, xi, normal_mu)
        delta = q * (self.tail_u + expected_excess)
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
            "normal_mu": normal_mu,
            "expected_excess": expected_excess,
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
        del z1_cls
        if phase == "pred":
            return self.predict_base(z1)
        if phase == "tail":
            if scaler is None:
                raise ValueError("scaler is required for tail prediction.")
            components = self.get_tail_prediction_components(z1, scaler)
            if t is not None:
                gate = (components["q"] > t).float()
                delta = gate * (self.tail_u + components["expected_excess"])
                return torch.clamp_min(components["y_hat_orig"] + delta, 0.0)
            return components["y_corr"]
        raise ValueError("phase not recognized")

    def weighted_reconstruction_loss(self, y_pred, y_true, val=False):
        loss_fn = self.loss_fun_val if val else self.loss_fun
        num_outputs = y_pred.size(-1)
        if num_outputs == 1:
            return loss_fn(y_pred[..., 0], y_true[..., 0])
        if num_outputs == 2:
            return self.args.yita * loss_fn(y_pred[..., 0], y_true[..., 0]) + (1 - self.args.yita) * loss_fn(
                y_pred[..., 1],
                y_true[..., 1],
            )
        channel_losses = [loss_fn(y_pred[..., idx], y_true[..., idx]) for idx in range(num_outputs)]
        return torch.stack(channel_losses).mean()

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
        if schedule.endswith("_then_joint"):
            schedule = schedule[: -len("_then_joint")]

        if schedule == "static":
            return {
                "schedule": schedule,
                "lambda_cls": base_lambda_cls,
                "lambda_gpd": base_lambda_gpd,
            }
        if epoch is None or total_epochs is None:
            raise ValueError(f"Epoch-aware tail schedule requires epoch and total_epochs: {schedule}")

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
            cls_multiplier = self._interpolate_segment(progress, warmup_end, mixed_end, 1.0, mixed_cls_end)
            gpd_multiplier = self._interpolate_segment(progress, warmup_end, mixed_end, 0.0, 1.0)
        else:
            cls_multiplier = self._interpolate_segment(progress, mixed_end, final_end, final_cls_start, final_cls_end)
            gpd_multiplier = 1.0

        return {
            "schedule": schedule,
            "lambda_cls": base_lambda_cls * cls_multiplier,
            "lambda_gpd": base_lambda_gpd * gpd_multiplier,
        }

    def get_default_tail_objective(self, epoch=None, total_epochs=None):
        tail_loss_weights = self.get_tail_loss_weights(epoch=epoch, total_epochs=total_epochs)
        return {
            "schedule": tail_loss_weights["schedule"],
            "lambda_cls": tail_loss_weights["lambda_cls"],
            "lambda_gpd": tail_loss_weights["lambda_gpd"],
            "lambda_mae": float(self.tail_mae_weight),
            "selection_metric_name": "corrected_mae",
            "tail_magnitude_mode": self.tail_magnitude_mode,
            "classifier_loss_type": self.tail_classifier_loss_type,
            "effective_pos_weight": float(self.tail_effective_pos_weight),
            "focal_gamma": float(self.tail_focal_gamma),
            "focal_alpha_pos": float(self.tail_focal_alpha_pos),
        }

    def focal_loss_with_logits(self, logit_q, indicator, gamma, alpha_pos):
        bce_loss = F.binary_cross_entropy_with_logits(logit_q, indicator, reduction="none")
        probs = torch.sigmoid(logit_q)
        p_t = indicator * probs + (1 - indicator) * (1 - probs)
        alpha_t = indicator * alpha_pos + (1 - indicator) * (1 - alpha_pos)
        focal_factor = (1 - p_t) ** gamma
        return (alpha_t * focal_factor * bce_loss).mean()

    def classification_loss(self, logit_q, indicator, objective_config=None):
        objective_config = objective_config or {}
        loss_type = objective_config.get("classifier_loss_type", self.tail_classifier_loss_type)
        if loss_type == "bce":
            return F.binary_cross_entropy_with_logits(logit_q, indicator)
        if loss_type == "weighted_bce":
            pos_weight = float(objective_config.get("effective_pos_weight", self.tail_effective_pos_weight))
            pos_weight_tensor = logit_q.new_tensor(pos_weight)
            return F.binary_cross_entropy_with_logits(logit_q, indicator, pos_weight=pos_weight_tensor)
        if loss_type == "focal":
            gamma = float(objective_config.get("focal_gamma", self.tail_focal_gamma))
            alpha_pos = float(objective_config.get("focal_alpha_pos", self.tail_focal_alpha_pos))
            return self.focal_loss_with_logits(logit_q, indicator, gamma=gamma, alpha_pos=alpha_pos)
        raise ValueError(f"Unsupported classifier loss type: {loss_type}")

    def gpd_loss(self, exceedance, indicator, xi, sigma):
        exceedance_mask = indicator > 0.5
        exceedance_count = int(exceedance_mask.sum().item())
        zero = sigma.new_zeros(())
        if exceedance_count == 0:
            return zero, {"exceedance_count": 0, "valid_exceedance_count": 0, "invalid_support_count": 0}

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

        loss = torch.log(safe_sigma[valid_mask]) + (1 / xi[valid_mask] + 1) * torch.log(
            torch.clamp(term[valid_mask], min=self.tail_eps)
        )
        return loss.mean(), {
            "exceedance_count": exceedance_count,
            "valid_exceedance_count": valid_exceedance_count,
            "invalid_support_count": invalid_support_count,
        }

    def point_excess_loss(self, exceedance, indicator, predicted_excess):
        exceedance_mask = indicator > 0.5
        exceedance_count = int(exceedance_mask.sum().item())
        zero = predicted_excess.new_zeros(())
        if exceedance_count == 0:
            return zero, {"exceedance_count": 0, "valid_exceedance_count": 0, "invalid_support_count": 0}
        return F.l1_loss(predicted_excess[exceedance_mask], exceedance[exceedance_mask]), {
            "exceedance_count": exceedance_count,
            "valid_exceedance_count": exceedance_count,
            "invalid_support_count": 0,
        }

    def normal_excess_loss(self, exceedance, indicator, mean_excess, std_excess):
        exceedance_mask = indicator > 0.5
        exceedance_count = int(exceedance_mask.sum().item())
        zero = std_excess.new_zeros(())
        if exceedance_count == 0:
            return zero, {"exceedance_count": 0, "valid_exceedance_count": 0, "invalid_support_count": 0}
        safe_std = torch.clamp(std_excess, min=self.tail_eps)
        residual = exceedance[exceedance_mask] - mean_excess[exceedance_mask]
        nll = (
            torch.log(safe_std[exceedance_mask])
            + 0.5 * (residual / safe_std[exceedance_mask]) ** 2
            + 0.5 * np.log(2.0 * np.pi)
        )
        return nll.mean(), {
            "exceedance_count": exceedance_count,
            "valid_exceedance_count": exceedance_count,
            "invalid_support_count": 0,
        }

    def magnitude_loss(self, exceedance, indicator, components):
        if self.tail_magnitude_mode == "gpd":
            return self.gpd_loss(exceedance, indicator, components["xi"], components["sigma"])
        if self.tail_magnitude_mode == "normal_excess":
            return self.normal_excess_loss(exceedance, indicator, components["normal_mu"], components["sigma"])
        if self.tail_magnitude_mode == "point_excess":
            return self.point_excess_loss(exceedance, indicator, components["expected_excess"])
        if self.tail_magnitude_mode in {"fixed_mean_excess", "threshold_only"}:
            exceedance_count = int((indicator > 0.5).sum().item())
            zero = components["q"].new_zeros(())
            return zero, {
                "exceedance_count": exceedance_count,
                "valid_exceedance_count": exceedance_count,
                "invalid_support_count": 0,
            }
        raise ValueError(f"Unsupported tail_magnitude_mode: {self.tail_magnitude_mode}")

    def loss(self, z1, z1_cls, evs, y_true, scaler, objective_config, phase, val=False):
        del z1_cls
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
                "selection_metric": pred_mae.item(),
                "selection_metric_name": "pred_mae",
                "exceedance_count": 0,
                "valid_exceedance_count": 0,
                "invalid_support_count": 0,
                "lambda_cls": 0.0,
                "lambda_gpd": 0.0,
                "lambda_mae": 0.0,
                "tail_schedule": "pred",
                "classifier_loss_type": "pred",
                "effective_pos_weight": 1.0,
            }
            return pred_mae, metrics

        if phase != "tail":
            raise ValueError("phase not recognized")

        tail_objective = objective_config or self.get_default_tail_objective()
        components = self.get_tail_prediction_components(z1, scaler)
        indicator, exceedance, _ = self.build_tail_targets(components["y_hat_orig"].detach(), y_true_orig)
        cls_loss = self.classification_loss(components["logit_q"], indicator, tail_objective)
        gpd_loss, gpd_stats = self.magnitude_loss(exceedance, indicator, components)
        corrected_mae = self.weighted_reconstruction_loss(components["y_corr"], y_true_orig, val=val)
        tail_loss = (
            tail_objective["lambda_cls"] * cls_loss
            + tail_objective["lambda_gpd"] * gpd_loss
            + tail_objective.get("lambda_mae", 0.0) * corrected_mae
        )
        selection_metric_name = tail_objective.get("selection_metric_name", "corrected_mae")
        selection_metric_map = {
            "loss": tail_loss.item(),
            "corrected_mae": corrected_mae.item(),
            "cls_loss": cls_loss.item(),
            "gpd_loss": gpd_loss.item(),
        }
        if selection_metric_name not in selection_metric_map:
            raise ValueError(f"Unsupported selection metric: {selection_metric_name}")
        metrics = {
            "pred_mae": corrected_mae.item(),
            "cls_loss": cls_loss.item(),
            "gpd_loss": gpd_loss.item(),
            "selection_metric": selection_metric_map[selection_metric_name],
            "selection_metric_name": selection_metric_name,
            "exceedance_count": gpd_stats["exceedance_count"],
            "valid_exceedance_count": gpd_stats["valid_exceedance_count"],
            "invalid_support_count": gpd_stats["invalid_support_count"],
            "lambda_cls": float(tail_objective["lambda_cls"]),
            "lambda_gpd": float(tail_objective["lambda_gpd"]),
            "lambda_mae": float(tail_objective.get("lambda_mae", 0.0)),
            "tail_schedule": tail_objective["schedule"],
            "tail_magnitude_mode": tail_objective.get("tail_magnitude_mode", self.tail_magnitude_mode),
            "classifier_loss_type": tail_objective.get("classifier_loss_type", self.tail_classifier_loss_type),
            "effective_pos_weight": float(tail_objective.get("effective_pos_weight", self.tail_effective_pos_weight)),
        }
        return tail_loss, metrics
