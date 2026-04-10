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
        residual_array = np.where(residual_array > 0, residual_array, np.nan)
        thresholds = np.zeros(residual_array.shape[1:], dtype=np.float32)
        for node_idx in range(residual_array.shape[1]):
            for flow_idx in range(residual_array.shape[2]):
                values = residual_array[:, node_idx, flow_idx]
                values = values[~np.isnan(values)]
                if values.size > 0:
                    thresholds[node_idx, flow_idx] = np.quantile(values, self.args.tail_threshold_q)
        thresholds = np.nan_to_num(thresholds, nan=0.0, posinf=0.0, neginf=0.0)
        thresholds = torch.from_numpy(thresholds).float().to(self.args.device)
        zero_threshold_count = int((thresholds == 0).sum().item())
        self.model.set_tail_thresholds(thresholds)
        self.logger.info(
            "Computed tail_u from the full training split with q={:.2f}. shape={} min={:.4f} max={:.4f} zero_entries={}".format(
                self.args.tail_threshold_q,
                tuple(self.model.tail_u.shape),
                self.model.tail_u.min().item(),
                self.model.tail_u.max().item(),
                zero_threshold_count,
            )
        )

    def _empty_metric_totals(self):
        return {
            "loss": 0.0,
            "pred_mae": 0.0,
            "cls_loss": 0.0,
            "gpd_loss": 0.0,
            "selection_mae": 0.0,
            "exceedance_count": 0,
            "valid_exceedance_count": 0,
            "invalid_support_count": 0,
        }

    def _finalize_epoch_stats(self, totals, num_batches):
        stats = {
            "loss": totals["loss"] / num_batches,
            "pred_mae": totals["pred_mae"] / num_batches,
            "cls_loss": totals["cls_loss"] / num_batches,
            "gpd_loss": totals["gpd_loss"] / num_batches,
            "selection_mae": totals["selection_mae"] / num_batches,
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

    def _log_epoch_stats(self, split, epoch, phase, stats):
        message = (
            f"*******{split} Epoch {epoch} [{phase}]: "
            f"loss={stats['loss']:.5f}, "
            f"mae={stats['pred_mae']:.5f}, "
            f"cls_loss={stats['cls_loss']:.5f}, "
            f"gpd_loss={stats['gpd_loss']:.5f}"
        )
        if phase == "tail":
            message += (
                f", exceedances={stats['exceedance_count']}, "
                f"valid_exceedances={stats['valid_exceedance_count']}, "
                f"invalid_support={stats['invalid_support_count']}, "
                f"invalid_rate={stats['invalid_support_rate']:.5f}"
            )
        self.logger.info(message)

    def train_epoch(self, epoch, phase):
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
                None,
                phase,
            )
            if not torch.isfinite(loss):
                raise ValueError(f"Encountered a non-finite {phase} training loss.")
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
            totals["selection_mae"] += metrics["selection_mae"]
            totals["exceedance_count"] += metrics["exceedance_count"]
            totals["valid_exceedance_count"] += metrics["valid_exceedance_count"]
            totals["invalid_support_count"] += metrics["invalid_support_count"]

        stats = self._finalize_epoch_stats(totals, self.train_per_epoch)
        self._log_epoch_stats("Train", epoch, phase, stats)
        return stats

    def val_epoch(self, epoch, phase):
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
                    None,
                    phase,
                    val=True,
                )
                if not torch.isnan(loss):
                    totals["loss"] += loss.item()
                    totals["pred_mae"] += metrics["pred_mae"]
                    totals["cls_loss"] += metrics["cls_loss"]
                    totals["gpd_loss"] += metrics["gpd_loss"]
                    totals["selection_mae"] += metrics["selection_mae"]
                    totals["exceedance_count"] += metrics["exceedance_count"]
                    totals["valid_exceedance_count"] += metrics["valid_exceedance_count"]
                    totals["invalid_support_count"] += metrics["invalid_support_count"]

        stats = self._finalize_epoch_stats(totals, len(val_dataloader))
        self._log_epoch_stats("Val", epoch, phase, stats)
        return stats

    def train_component(self, params_to_train, params_to_freeze, phase, esp):
        self._set_trainable_params(params_to_train, params_to_freeze)

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

        for epoch in range(1, self.args.epochs + 1):
            train_stats = self.train_epoch(epoch, phase)
            val_stats = self.val_epoch(epoch, phase)

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

            selection_metric = val_stats["selection_mae"]
            self.logger.info(
                "Selection metric [{}] at epoch {}: {:.5f}".format(
                    phase,
                    epoch,
                    selection_metric,
                )
            )
            if selection_metric < best_metric:
                best_metric = selection_metric
                best_epoch = epoch
                not_improved_count = 0
                best_save_dict = self._save_checkpoint(epoch, phase)
            else:
                not_improved_count += 1

            if self.args.early_stop and not_improved_count >= esp:
                self.logger.info(
                    "Validation {} MAE did not improve for {} epochs. Ending {} training.".format(
                        phase,
                        esp,
                        phase,
                    )
                )
                break

        if best_save_dict is None:
            best_save_dict = self._save_checkpoint(epoch, phase)

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
        self.logger.info("== Test results.")
        test_results = self.test(
            self.model,
            self.test_loader,
            self.scaler,
            self.graph,
            self.logger,
            self.args,
            phase,
        )
        results = {
            "best_val_metric": best_metric,
            "best_val_epoch": best_epoch,
            "test_results": test_results,
        }
        self.plot_losses(history, phase)
        return results

    def train(self):
        self._save_source_files()
        pred_params, classifier_params, gpd_params = get_model_params_grouped(self.model)

        pred_results = None
        if self.args.load_path is None:
            pred_results = self.train_component(
                pred_params,
                classifier_params + gpd_params,
                "pred",
                esp=self.args.early_stop_patience,
            )
            self._load_checkpoint(self.best_path, strict=False, checkpoint_label="best pred checkpoint")
        else:
            self.logger.info("Skipping pred phase because load_path was provided.")

        self.compute_tail_thresholds()
        pred_params, classifier_params, gpd_params = get_model_params_grouped(self.model)
        tail_results = self.train_component(
            classifier_params + gpd_params,
            pred_params,
            "tail",
            esp=self.args.early_stop_patience,
        )
        return {
            "pred": pred_results,
            "tail": tail_results,
        }

    def plot_losses(self, history, phase):
        plt.figure(figsize=(12, 8))
        plt.plot(history["train_loss"], label="Train Loss")
        plt.plot(history["val_loss"], label="Val Loss")
        plt.plot(history["train_mae"], label="Train MAE")
        plt.plot(history["val_mae"], label="Val MAE")
        if phase == "tail":
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
