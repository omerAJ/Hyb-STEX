import os
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from lib.logger import (
    PD_Stats,
    get_logger,
)
from lib.metrics import test_metrics
from lib.utils import (
    get_log_dir,
    get_model_params,
)
from model.parameter_groups import format_param_group_counts, get_model_params_grouped


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Trainer(object):
    def __init__(self, model, optimizer, dataloader, graph, args):
        super(Trainer, self).__init__()
        self.model = model
        self.graph = graph
        self.args = args

        if isinstance(self.model, torch.nn.Module):
            with torch.no_grad():
                dummy_view = self._get_dummy_input(dataloader["val"], args)
                _, _ = self.model(dummy_view, self.graph)

        self.optimizer = optimizer
        self.base_optimizer_lrs = [group["lr"] for group in self.optimizer.param_groups]
        self.train_loader = dataloader["train"]
        self.val_loader = dataloader["val"]
        self.test_loader = dataloader["test"]
        self.scaler = dataloader["scaler"]
        self.train_per_epoch = len(self.train_loader)
        self.val_per_epoch = len(self.val_loader) if self.val_loader is not None else 0

        args.log_dir = get_log_dir(args)
        if not os.path.isdir(args.log_dir) and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.log_dir, debug=args.debug)
        self.best_path = os.path.join(self.args.log_dir, "best_model.pth")
        self.logs_dir = self.args.log_dir
        self.num_params = count_parameters(self.model)

        self.training_stats = PD_Stats(
            os.path.join(args.log_dir, "stats.pkl"),
            ["epoch", "train_loss", "val_loss"],
        )
        self.logger.info("\nModel has {} M trainable parameters".format(self.num_params / (1e6)))
        self.logger.info("Experiment log path in: {}".format(args.log_dir))
        self.logger.info("Experiment configs are: {}".format(args))
        self.logger.info("\nModel has {} M trainable parameters".format(self.num_params / (1e6)))
        self.logger.info(format_param_group_counts(self.model))
        self.tail_target_stats = {
            "positive_count": 0,
            "negative_count": 0,
            "exceedance_rate": 0.0,
            "base_pos_weight": 1.0,
            "effective_pos_weight": 1.0,
            "zero_threshold_count": 0,
        }

        if args.load_path is not None:
            self._load_checkpoint(args.load_path, strict=False, checkpoint_label="external base checkpoint")

    def _get_dummy_input(self, dataloader, args):
        for batch in dataloader:
            return batch[0].to(args.device)
        raise RuntimeError("Unable to build dummy input because the dataloader is empty.")

    def _save_source_files(self):
        current_directory = os.path.dirname(os.path.abspath(__file__))
        models_file_path = os.path.join(current_directory, "models.py")
        layers_file_path = os.path.join(current_directory, "layers.py")
        trainer_file_path = os.path.join(current_directory, "trainer.py")
        main_file_path = os.path.join(os.path.dirname(current_directory), "main.py")
        save_dir = self.logs_dir
        shutil.copy(models_file_path, save_dir)
        shutil.copy(layers_file_path, save_dir)
        shutil.copy(trainer_file_path, save_dir)
        shutil.copy(main_file_path, save_dir)
        self.logger.info("Model code files saved in: {}".format(save_dir))

    def _load_checkpoint(self, path, strict=False, checkpoint_label="checkpoint"):
        checkpoint = torch.load(path, map_location=torch.device(self.args.device))
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        msg = self.model.load_state_dict(state_dict, strict=strict)
        self.logger.info("Loaded {} from {}".format(checkpoint_label, path))
        self.logger.info("Checkpoint load message: {}".format(msg))
        return checkpoint

    def _save_checkpoint(self, epoch, phase):
        save_dict = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        self.best_path = os.path.join(self.args.log_dir, f"best_model_{phase}.pth")
        if not self.args.debug:
            torch.save(save_dict, self.best_path)
            self.logger.info("**************Current best model saved to {}".format(self.best_path))
        return save_dict

    def _set_trainable_params(self, params_to_train, params_to_freeze):
        for param in params_to_train:
            param.requires_grad = True
        for param in params_to_freeze:
            param.requires_grad = False

    def _set_optimizer_group_lrs(self, pred_scale=1.0, classifier_scale=1.0, gpd_scale=1.0):
        lr_scales = [pred_scale, classifier_scale, gpd_scale]
        for group, base_lr, scale in zip(self.optimizer.param_groups, self.base_optimizer_lrs, lr_scales):
            group["lr"] = base_lr * scale

    def _restore_optimizer_group_lrs(self):
        for group, base_lr in zip(self.optimizer.param_groups, self.base_optimizer_lrs):
            group["lr"] = base_lr

    def _uses_two_stage_tail(self):
        return self.args.tail_schedule in {
            "cls_then_gpd_freeze",
            "cls_then_gpd_light_cls",
            "cls_then_gpd_freeze_then_joint",
            "cls_then_gpd_light_cls_then_joint",
        }

    def _uses_joint_refine_tail(self):
        return self.args.tail_schedule in {
            "static_then_joint",
            "cls_then_gpd_freeze_then_joint",
            "cls_then_gpd_light_cls_then_joint",
        }

    def _get_full_train_loader(self):
        return DataLoader(
            self.train_loader.dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            drop_last=False,
        )

    def compute_tail_thresholds(self):
        self.model.eval()
        positive_residuals = []
        full_train_loader = self._get_full_train_loader()
        with torch.no_grad():
            for data, target, _, _ in full_train_loader:
                repr1, repr1_cls = self.model(data, self.graph)
                y_hat = self.model.predict(repr1, repr1_cls, "pred")
                y_hat_orig = self.scaler.inverse_transform(y_hat)
                y_true_orig = self.scaler.inverse_transform(target)
                residual = torch.clamp(y_true_orig - y_hat_orig, min=0.0)
                positive_residuals.append(residual.squeeze(1).cpu().numpy())

        residual_array = np.concatenate(positive_residuals, axis=0)
        positive_only_residuals = np.where(residual_array > 0, residual_array, np.nan)
        thresholds = np.zeros(residual_array.shape[1:], dtype=np.float32)
        for node_idx in range(positive_only_residuals.shape[1]):
            for flow_idx in range(positive_only_residuals.shape[2]):
                values = positive_only_residuals[:, node_idx, flow_idx]
                values = values[~np.isnan(values)]
                if values.size > 0:
                    thresholds[node_idx, flow_idx] = np.quantile(values, self.args.tail_threshold_q)
        thresholds = np.nan_to_num(thresholds, nan=0.0, posinf=0.0, neginf=0.0)
        thresholds = torch.from_numpy(thresholds).float().to(self.args.device)
        zero_threshold_count = int((thresholds == 0).sum().item())
        self.model.set_tail_thresholds(thresholds)
        exceedance_indicator = residual_array > thresholds.unsqueeze(0).cpu().numpy()
        positive_count = int(exceedance_indicator.sum())
        total_count = int(exceedance_indicator.size)
        negative_count = max(total_count - positive_count, 0)
        exceedance_rate = (positive_count / total_count) if total_count > 0 else 0.0
        base_pos_weight = (negative_count / positive_count) if positive_count > 0 else 1.0
        effective_pos_weight = 1.0
        if self.args.tail_classifier_loss_type == "weighted_bce" and positive_count > 0:
            effective_pos_weight = base_pos_weight * float(self.args.tail_pos_weight_multiplier)
        self.tail_target_stats = {
            "positive_count": positive_count,
            "negative_count": negative_count,
            "exceedance_rate": exceedance_rate,
            "base_pos_weight": float(base_pos_weight),
            "effective_pos_weight": float(effective_pos_weight),
            "zero_threshold_count": zero_threshold_count,
        }
        self.model.set_tail_classifier_stats(self.tail_target_stats)
        self.logger.info(
            "Computed tail_u from the full training split with q={:.2f}. shape={} min={:.4f} max={:.4f} zero_entries={} exceedance_rate={:.6f} base_pos_weight={:.4f} effective_pos_weight={:.4f}".format(
                self.args.tail_threshold_q,
                tuple(self.model.tail_u.shape),
                self.model.tail_u.min().item(),
                self.model.tail_u.max().item(),
                zero_threshold_count,
                exceedance_rate,
                base_pos_weight,
                effective_pos_weight,
            )
        )
        return self.tail_target_stats

    def _empty_metric_totals(self):
        return {
            "loss": 0.0,
            "pred_mae": 0.0,
            "cls_loss": 0.0,
            "gpd_loss": 0.0,
            "selection_metric": 0.0,
            "exceedance_count": 0,
            "valid_exceedance_count": 0,
            "invalid_support_count": 0,
        }

    def _build_tail_objective(self, epoch, phase_label, total_epochs=None):
        if phase_label == "tail":
            objective = self.model.get_default_tail_objective(
                epoch=epoch,
                total_epochs=self.args.epochs if total_epochs is None else total_epochs,
            )
            objective["selection_metric_name"] = "corrected_mae"
            return objective
        if phase_label == "tail_stage1":
            return {
                "schedule": self.args.tail_schedule,
                "lambda_cls": float(self.args.tail_lambda_cls),
                "lambda_gpd": 0.0,
                "lambda_mae": 0.0,
                "selection_metric_name": "cls_loss",
                "classifier_loss_type": self.args.tail_classifier_loss_type,
                "effective_pos_weight": float(self.tail_target_stats["effective_pos_weight"]),
                "focal_gamma": float(self.args.tail_focal_gamma),
                "focal_alpha_pos": float(self.args.tail_focal_alpha_pos),
            }
        if phase_label == "tail_stage2_freeze":
            return {
                "schedule": self.args.tail_schedule,
                "lambda_cls": 0.0,
                "lambda_gpd": float(self.args.tail_lambda_gpd),
                "lambda_mae": float(self.args.tail_mae_weight),
                "selection_metric_name": "corrected_mae",
                "classifier_loss_type": self.args.tail_classifier_loss_type,
                "effective_pos_weight": float(self.tail_target_stats["effective_pos_weight"]),
                "focal_gamma": float(self.args.tail_focal_gamma),
                "focal_alpha_pos": float(self.args.tail_focal_alpha_pos),
            }
        if phase_label == "tail_stage2_light_cls":
            return {
                "schedule": self.args.tail_schedule,
                "lambda_cls": 0.1 * float(self.args.tail_lambda_cls),
                "lambda_gpd": float(self.args.tail_lambda_gpd),
                "lambda_mae": float(self.args.tail_mae_weight),
                "selection_metric_name": "corrected_mae",
                "classifier_loss_type": self.args.tail_classifier_loss_type,
                "effective_pos_weight": float(self.tail_target_stats["effective_pos_weight"]),
                "focal_gamma": float(self.args.tail_focal_gamma),
                "focal_alpha_pos": float(self.args.tail_focal_alpha_pos),
            }
        if phase_label == "tail_stage3_joint":
            return {
                "schedule": self.args.tail_schedule,
                "lambda_cls": float(getattr(self.args, "joint_refine_lambda_cls", 0.05)),
                "lambda_gpd": float(getattr(self.args, "joint_refine_lambda_gpd", self.args.tail_lambda_gpd)),
                "lambda_mae": float(getattr(self.args, "joint_refine_lambda_mae", self.args.tail_mae_weight)),
                "selection_metric_name": "corrected_mae",
                "classifier_loss_type": self.args.tail_classifier_loss_type,
                "effective_pos_weight": float(self.tail_target_stats["effective_pos_weight"]),
                "focal_gamma": float(self.args.tail_focal_gamma),
                "focal_alpha_pos": float(self.args.tail_focal_alpha_pos),
            }
        return None

    def _finalize_epoch_stats(self, totals, num_batches, selection_metric_name):
        stats = {
            "loss": totals["loss"] / num_batches,
            "pred_mae": totals["pred_mae"] / num_batches,
            "cls_loss": totals["cls_loss"] / num_batches,
            "gpd_loss": totals["gpd_loss"] / num_batches,
            "selection_metric": totals["selection_metric"] / num_batches,
            "selection_metric_name": selection_metric_name,
            "exceedance_count": totals["exceedance_count"],
            "valid_exceedance_count": totals["valid_exceedance_count"],
            "invalid_support_count": totals["invalid_support_count"],
        }
        stats["invalid_support_rate"] = (
            stats["invalid_support_count"] / stats["exceedance_count"]
            if stats["exceedance_count"] > 0
            else 0.0
        )
        return stats

    def _log_epoch_stats(self, split, epoch, phase, stats, objective_config=None):
        message = (
            f"*******{split} Epoch {epoch} [{phase}]: "
            f"loss={stats['loss']:.5f}, "
            f"mae={stats['pred_mae']:.5f}, "
            f"cls_loss={stats['cls_loss']:.5f}, "
            f"gpd_loss={stats['gpd_loss']:.5f}"
        )
        if phase.startswith("tail"):
            message += (
                f", exceedances={stats['exceedance_count']}, "
                f"valid_exceedances={stats['valid_exceedance_count']}, "
                f"invalid_support={stats['invalid_support_count']}, "
                f"invalid_rate={stats['invalid_support_rate']:.5f}"
            )
            if objective_config is not None:
                message += (
                    f", lambda_cls={objective_config['lambda_cls']:.5f}, "
                    f"lambda_gpd={objective_config['lambda_gpd']:.5f}, "
                    f"lambda_mae={objective_config.get('lambda_mae', 0.0):.5f}, "
                    f"schedule={objective_config['schedule']}, "
                    f"clf_loss={objective_config.get('classifier_loss_type', 'bce')}, "
                    f"selection={stats['selection_metric_name']}"
                )
        self.logger.info(message)

    def train_epoch(self, epoch, model_phase, phase_label, objective_config=None):
        self.model.train()
        totals = self._empty_metric_totals()

        for data, target, evs, _ in self.train_loader:
            self.optimizer.zero_grad()
            repr1, repr1_cls = self.model(data, self.graph)
            loss, metrics = self.model.loss(
                repr1,
                repr1_cls,
                evs,
                target,
                self.scaler,
                objective_config,
                model_phase,
            )
            if not torch.isfinite(loss):
                raise ValueError(f"Encountered a non-finite {phase_label} training loss.")
            loss.backward()

            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    get_model_params([self.model]),
                    self.args.max_grad_norm,
                )
            self.optimizer.step()

            totals["loss"] += loss.item()
            totals["pred_mae"] += metrics["pred_mae"]
            totals["cls_loss"] += metrics["cls_loss"]
            totals["gpd_loss"] += metrics["gpd_loss"]
            totals["selection_metric"] += metrics["selection_metric"]
            totals["exceedance_count"] += metrics["exceedance_count"]
            totals["valid_exceedance_count"] += metrics["valid_exceedance_count"]
            totals["invalid_support_count"] += metrics["invalid_support_count"]

        selection_metric_name = "pred_mae" if objective_config is None else objective_config["selection_metric_name"]
        stats = self._finalize_epoch_stats(totals, self.train_per_epoch, selection_metric_name)
        self._log_epoch_stats("Train", epoch, phase_label, stats, objective_config=objective_config)
        return stats

    def val_epoch(self, epoch, model_phase, phase_label, objective_config=None):
        self.model.eval()
        totals = self._empty_metric_totals()
        val_dataloader = self.val_loader if self.val_loader is not None else self.test_loader

        with torch.no_grad():
            for data, target, evs, _ in val_dataloader:
                repr1, repr1_cls = self.model(data, self.graph)
                loss, metrics = self.model.loss(
                    repr1,
                    repr1_cls,
                    evs,
                    target,
                    self.scaler,
                    objective_config,
                    model_phase,
                    val=True,
                )
                if not torch.isnan(loss):
                    totals["loss"] += loss.item()
                    totals["pred_mae"] += metrics["pred_mae"]
                    totals["cls_loss"] += metrics["cls_loss"]
                    totals["gpd_loss"] += metrics["gpd_loss"]
                    totals["selection_metric"] += metrics["selection_metric"]
                    totals["exceedance_count"] += metrics["exceedance_count"]
                    totals["valid_exceedance_count"] += metrics["valid_exceedance_count"]
                    totals["invalid_support_count"] += metrics["invalid_support_count"]

        selection_metric_name = "pred_mae" if objective_config is None else objective_config["selection_metric_name"]
        stats = self._finalize_epoch_stats(totals, len(val_dataloader), selection_metric_name)
        self._log_epoch_stats("Val", epoch, phase_label, stats, objective_config=objective_config)
        return stats

    def train_component(
        self,
        params_to_train,
        params_to_freeze,
        model_phase,
        phase_label,
        esp,
        run_test=True,
        test_phase=None,
        max_epochs=None,
        lr_scales=None,
    ):
        self._set_trainable_params(params_to_train, params_to_freeze)
        if lr_scales is not None:
            self._set_optimizer_group_lrs(
                pred_scale=lr_scales.get("pred", 1.0),
                classifier_scale=lr_scales.get("classifier", 1.0),
                gpd_scale=lr_scales.get("gpd", 1.0),
            )
        total_epochs = self.args.epochs if max_epochs is None else max_epochs

        history = {
            "train_loss": [],
            "val_loss": [],
            "train_mae": [],
            "val_mae": [],
            "train_cls_loss": [],
            "val_cls_loss": [],
            "train_gpd_loss": [],
            "val_gpd_loss": [],
        }
        best_metric = float("inf")
        best_epoch = 0
        not_improved_count = 0
        best_save_dict = None
        start_time = time.time()

        for epoch in range(1, total_epochs + 1):
            objective_config = (
                self._build_tail_objective(epoch, phase_label, total_epochs=total_epochs)
                if model_phase == "tail"
                else None
            )
            train_stats = self.train_epoch(epoch, model_phase, phase_label, objective_config=objective_config)
            val_stats = self.val_epoch(epoch, model_phase, phase_label, objective_config=objective_config)

            history["train_loss"].append(train_stats["loss"])
            history["val_loss"].append(val_stats["loss"])
            history["train_mae"].append(train_stats["pred_mae"])
            history["val_mae"].append(val_stats["pred_mae"])
            history["train_cls_loss"].append(train_stats["cls_loss"])
            history["val_cls_loss"].append(val_stats["cls_loss"])
            history["train_gpd_loss"].append(train_stats["gpd_loss"])
            history["val_gpd_loss"].append(val_stats["gpd_loss"])

            if not self.args.debug:
                self.training_stats.update((epoch, train_stats["loss"], val_stats["loss"]))

            selection_metric = val_stats["selection_metric"]
            self.logger.info(
                "Selection metric [{}:{}] at epoch {}: {:.5f}".format(
                    phase_label,
                    val_stats["selection_metric_name"],
                    epoch,
                    selection_metric,
                )
            )
            if selection_metric < best_metric:
                best_metric = selection_metric
                best_epoch = epoch
                not_improved_count = 0
                best_save_dict = self._save_checkpoint(epoch, phase_label)
            else:
                not_improved_count += 1

            if self.args.early_stop and not_improved_count >= esp:
                self.logger.info(
                    "Validation {} {} did not improve for {} epochs. Ending {} training.".format(
                        phase_label,
                        val_stats["selection_metric_name"],
                        esp,
                        phase_label,
                    )
                )
                break

        if best_save_dict is None:
            best_save_dict = self._save_checkpoint(epoch, phase_label)

        training_time = time.time() - start_time
        self.logger.info(
            "== Training finished.\n"
            "Total training time: {:.2f} min\t"
            "best selection metric: {:.4f}\t"
            "best epoch: {}\t".format(
                (training_time / 60),
                best_metric,
                best_epoch,
            )
        )
        state_dict = best_save_dict if self.args.debug else torch.load(
            self.best_path,
            map_location=torch.device(self.args.device),
        )
        self.model.load_state_dict(state_dict["model"])
        self._restore_optimizer_group_lrs()
        results = {
            "best_val_metric": best_metric,
            "best_val_epoch": best_epoch,
            "selection_metric_name": val_stats["selection_metric_name"],
            "test_results": None,
            "test_metrics": None,
        }
        if run_test:
            self.logger.info("== Test results.")
            test_results = self.test(
                self.model,
                self.test_loader,
                self.scaler,
                self.graph,
                self.logger,
                self.args,
                test_phase or model_phase,
            )
            results["test_results"] = test_results
            results["test_metrics"] = self.format_test_results(test_results)
        self.plot_losses(history, phase_label)
        return results

    def train(self):
        self._save_source_files()
        pred_params, classifier_params, gpd_params = get_model_params_grouped(self.model)
        training_recipe = getattr(self.args, "training_recipe", "full")
        pred_results = None

        if training_recipe not in {"full", "pred_only", "tail_only"}:
            raise ValueError(f"Unsupported training_recipe: {training_recipe}")

        if training_recipe in {"full", "pred_only"} and self.args.load_path is None:
            pred_results = self.train_component(
                pred_params,
                classifier_params + gpd_params,
                "pred",
                "pred",
                esp=self.args.early_stop_patience,
                run_test=True,
                test_phase="pred",
            )
            self._load_checkpoint(self.best_path, strict=False, checkpoint_label="best pred checkpoint")
            if training_recipe == "pred_only":
                return {"pred": pred_results, "tail": None}
        elif training_recipe == "pred_only":
            raise ValueError("pred_only training requires training a fresh base predictor. Remove load_path.")
        else:
            self.logger.info("Skipping pred phase because load_path was provided or tail_only recipe was selected.")

        if training_recipe == "tail_only" and self.args.load_path is None:
            raise ValueError("tail_only training requires load_path to point to a base predictor checkpoint.")

        self.compute_tail_thresholds()
        pred_params, classifier_params, gpd_params = get_model_params_grouped(self.model)
        if self._uses_two_stage_tail():
            tail_results = self.train_tail_two_stage(pred_params, classifier_params, gpd_params)
        elif self._uses_joint_refine_tail():
            tail_results = self.train_tail_static_then_joint(pred_params, classifier_params, gpd_params)
        else:
            tail_results = self.train_component(
                classifier_params + gpd_params,
                pred_params,
                "tail",
                "tail",
                esp=self.args.early_stop_patience,
                run_test=True,
                test_phase="tail",
            )
            tail_results["stage1"] = None
            tail_results["stage2"] = None
            tail_results["stage3"] = None
            tail_results["tail_target_stats"] = dict(self.tail_target_stats)
        return {"pred": pred_results, "tail": tail_results}

    def train_tail_two_stage(self, pred_params, classifier_params, gpd_params):
        stage1_results = self.train_component(
            classifier_params,
            pred_params + gpd_params,
            "tail",
            "tail_stage1",
            esp=self.args.early_stop_patience,
            run_test=False,
            test_phase="tail",
        )
        self._load_checkpoint(self.best_path, strict=False, checkpoint_label="best tail stage1 checkpoint")
        stage2_phase = "tail_stage2_freeze" if self.args.tail_schedule == "cls_then_gpd_freeze" else "tail_stage2_light_cls"
        if stage2_phase == "tail_stage2_freeze":
            params_to_train = gpd_params
            params_to_freeze = pred_params + classifier_params
        else:
            params_to_train = classifier_params + gpd_params
            params_to_freeze = pred_params
        stage2_results = self.train_component(
            params_to_train,
            params_to_freeze,
            "tail",
            stage2_phase,
            esp=self.args.early_stop_patience,
            run_test=not self._uses_joint_refine_tail(),
            test_phase="tail",
        )
        tail_results = dict(stage2_results)
        stage3_results = None
        if self._uses_joint_refine_tail():
            stage3_results = self.train_joint_refine_stage(pred_params, classifier_params, gpd_params)
            tail_results = dict(stage3_results)
        tail_results["stage1"] = stage1_results
        tail_results["stage2"] = stage2_results
        tail_results["stage3"] = stage3_results
        tail_results["tail_target_stats"] = dict(self.tail_target_stats)
        return tail_results

    def train_tail_static_then_joint(self, pred_params, classifier_params, gpd_params):
        stage2_results = self.train_component(
            classifier_params + gpd_params,
            pred_params,
            "tail",
            "tail",
            esp=self.args.early_stop_patience,
            run_test=False,
            test_phase="tail",
        )
        stage3_results = self.train_joint_refine_stage(pred_params, classifier_params, gpd_params)
        tail_results = dict(stage3_results)
        tail_results["stage1"] = None
        tail_results["stage2"] = stage2_results
        tail_results["stage3"] = stage3_results
        tail_results["tail_target_stats"] = dict(self.tail_target_stats)
        return tail_results

    def train_joint_refine_stage(self, pred_params, classifier_params, gpd_params):
        if getattr(self.args, "joint_refine_recompute_tail_u", False):
            self.compute_tail_thresholds()
        return self.train_component(
            pred_params + classifier_params + gpd_params,
            [],
            "tail",
            "tail_stage3_joint",
            esp=int(getattr(self.args, "joint_refine_early_stop_patience", self.args.early_stop_patience)),
            run_test=True,
            test_phase="tail",
            max_epochs=int(getattr(self.args, "joint_refine_epochs", self.args.epochs)),
            lr_scales={
                "pred": float(getattr(self.args, "joint_refine_pred_lr_scale", 0.1)),
                "classifier": float(getattr(self.args, "joint_refine_classifier_lr_scale", 1.0)),
                "gpd": float(getattr(self.args, "joint_refine_gpd_lr_scale", 1.0)),
            },
        )

    def plot_losses(self, history, phase):
        plt.figure(figsize=(12, 8))
        plt.plot(history["train_loss"], label="Train Loss")
        plt.plot(history["val_loss"], label="Val Loss")
        plt.plot(history["train_mae"], label="Train MAE")
        plt.plot(history["val_mae"], label="Val MAE")
        if phase.startswith("tail"):
            plt.plot(history["train_cls_loss"], label="Train BCE")
            plt.plot(history["val_cls_loss"], label="Val BCE")
            plt.plot(history["train_gpd_loss"], label="Train GPD")
            plt.plot(history["val_gpd_loss"], label="Val GPD")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"Losses [{phase}]")
        plt.legend()
        plt.savefig(os.path.join(self.args.log_dir, f"losses_{phase}.png"))
        plt.close()

    @staticmethod
    def format_test_results(test_results):
        if test_results is None:
            return None
        return {
            "inflow": {
                "mae": float(test_results[0][0]),
                "eee": float(test_results[0][1]),
            },
            "outflow": {
                "mae": float(test_results[1][0]),
                "eee": float(test_results[1][1]),
            },
        }

    @staticmethod
    def test(model, dataloader, scaler, graph, logger, args, phase):
        model.eval()
        y_pred = []
        y_true = []
        evs_true = []

        with torch.no_grad():
            for data, target, evs, _ in dataloader:
                repr1, repr1_cls = model(data, graph)
                pred_output = model.predict(repr1, repr1_cls, phase, scaler=scaler if phase == "tail" else None)
                if phase == "pred":
                    pred_output = scaler.inverse_transform(pred_output)
                y_true.append(scaler.inverse_transform(target))
                y_pred.append(pred_output)
                evs_true.append(evs)

        y_true = torch.cat(y_true, dim=0).cpu()
        y_pred = torch.cat(y_pred, dim=0).cpu()
        evs_true = torch.cat(evs_true, dim=0).cpu()

        test_results = []
        mae, eee = test_metrics(y_pred[..., 0], y_true[..., 0], evs=evs_true[..., 0])
        logger.info("INFLOW, MAE: {:.2f}, EEE: {:.4f}".format(mae, eee))
        test_results.append([mae, eee])
        mae, eee = test_metrics(y_pred[..., 1], y_true[..., 1], evs=evs_true[..., 1])
        logger.info("OUTFLOW, MAE: {:.2f}, EEE: {:.4f}".format(mae, eee))
        test_results.append([mae, eee])
        return np.stack(test_results, axis=0)


def plot_cm(pred, true, gt=None):
    if gt is not None:
        mask_value = 5.0
        mask = torch.gt(gt, mask_value).cpu()
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    pred = pred.cpu().numpy()
    true = true.cpu().numpy()
    evs_pred_binary = (pred >= 0.5).astype(int)

    evs_true_flat = true.flatten()
    evs_pred_flat = evs_pred_binary.flatten()
    conf_matrix = confusion_matrix(evs_true_flat, evs_pred_flat)
    return conf_matrix
