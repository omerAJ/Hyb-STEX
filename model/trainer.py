import os
import time
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
    
# torch.autograd.set_detect_anomaly(True)
import matplotlib.pyplot as plt

from lib.logger import (
    get_logger, 
    PD_Stats, 
)
from lib.utils import (
    get_log_dir, 
    get_model_params, 
    dwa,  
)
from lib.metrics import corrected_event_metrics, test_metrics

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def is_residual_bias_param(name):
    return (
        name == "learnable_vectors_bias"
        or name == "learnable_vectors_event_bias"
        or name == "node_bias"
        or name == "flow_bias"
        or name.startswith("ff_key_projection_bias.")
        or name.startswith("ff_key_projection_event_bias.")
    )


def is_active_residual_bias_param(name, args):
    mode = getattr(args, "ablation_mode", "original")
    if mode == "node_bias_only":
        return name == "node_bias"
    if mode == "flow_bias_only":
        return name == "flow_bias"
    if mode in {
        "dual_event_residual",
        "event_weighted_dual_residual",
        "event_weighted_dual_ungated",
    }:
        return (
            name == "learnable_vectors_bias"
            or name == "learnable_vectors_event_bias"
            or name.startswith("ff_key_projection_bias.")
            or name.startswith("ff_key_projection_event_bias.")
        )
    return (
        name == "learnable_vectors_bias"
        or name.startswith("ff_key_projection_bias.")
    )


def get_model_params_grouped(model, args=None):
    if args is None:
        args = getattr(model, "args", None)
    bias_param_scope = getattr(args, "bias_param_scope", "legacy")
    pred_params = []
    classifier_params = []
    bias_params = []
    for name, param in model.named_parameters():
        if 'cls' in name:
            classifier_params.append(param)
        elif bias_param_scope == "head_only" and is_active_residual_bias_param(name, args):
            bias_params.append(param)
        elif bias_param_scope != "head_only" and "bias" in name:
            bias_params.append(param)
        else:
            pred_params.append(param)
    return pred_params, classifier_params, bias_params

def uses_ungated_bias_ablation(args):
    return getattr(args, "ablation_mode", "original") in {
        "ungated_bias",
        "event_weighted_ungated_end_to_end",
        "event_weighted_ungated_residual",
        "event_weighted_dual_ungated",
        "node_bias_only",
        "flow_bias_only",
    }


def uses_expanded_base_additive_ablation(args):
    return getattr(args, "ablation_mode", "original") == "expanded_base_additive"


def build_evaluation_details(prediction, target, event, capture_predictions=False, valid_min=5.0):
    """Build corrected per-flow evaluation details and optional CPU artifacts."""
    if prediction.shape != target.shape or prediction.shape != event.shape:
        raise ValueError("prediction, target, and event must share a shape")
    if prediction.ndim < 1:
        raise ValueError("prediction, target, and event must include a flow dimension")

    prediction_cpu = prediction.detach().to(device="cpu")
    target_cpu = target.detach().to(device="cpu")
    event_cpu = event.detach().to(device="cpu")
    if event_cpu.dtype != torch.bool:
        integer_dtypes = {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }
        is_real_numeric = not event_cpu.is_complex() and (
            event_cpu.is_floating_point() or event_cpu.dtype in integer_dtypes
        )
        is_binary = is_real_numeric and bool(
            torch.all((event_cpu == 0) | (event_cpu == 1)).item()
        )
        if not is_binary:
            raise ValueError("event must contain only bool or exact binary 0/1 values")
    valid = target_cpu > valid_min
    # Legacy labels can include invalid points. Evaluation details always use the
    # corrected valid/event partition, while legacy headline metrics remain intact.
    corrected_event = event_cpu.to(dtype=torch.bool) & valid

    details = [
        corrected_event_metrics(
            prediction_cpu[..., flow_idx],
            target_cpu[..., flow_idx],
            corrected_event[..., flow_idx],
            valid_min=valid_min,
        )
        for flow_idx in range(prediction_cpu.shape[-1])
    ]
    artifacts = None
    if capture_predictions:
        artifacts = {
            "prediction": prediction_cpu.to(dtype=torch.float32),
            "target": target_cpu.to(dtype=torch.float32),
            "event": corrected_event.to(dtype=torch.float32),
            "valid": valid.to(dtype=torch.float32),
        }
    return details, artifacts


def phase_index(name):
    phases = {"pred": 0, "cls": 1, "bias": 2, "pred_2": 3}
    if name not in phases:
        raise ValueError(f"Unsupported training phase: {name}")
    return phases[name]

class Trainer(object):
    def __init__(self, model, optimizer, dataloader, graph, args):
        super(Trainer, self).__init__()
        self.model = model
        self.graph = graph
        self.args = args

        if isinstance(self.model, torch.nn.Module):  # Check if it's a PyTorch module
            with torch.no_grad():  # Temporarily disable gradient calculations
                dummy_view = self._get_dummy_input(dataloader['val'], args)  
                _, _ = self.model(dummy_view, self.graph)  # Trigger initialization
                
        
        def rename_keys(state_dict):
            renamed_state_dict = {}
            for key in state_dict.keys():
                # Example renaming pattern, adjust according to your needs
                new_key = key.replace('attention1', 'attentive_fuse.attention1').replace('attention2', 'attentive_fuse.attention2')
                renamed_state_dict[new_key] = state_dict[key]
            return renamed_state_dict
        print("dummy forward pass done.")
        path_to_load = args.load_path
        if path_to_load is not None:
            state_dict = torch.load(
                path_to_load, map_location=torch.device(args.device))['model']
            msg = self.model.load_state_dict(state_dict, strict=False) 
            print("loading pretrained model from: ", path_to_load)
            print("\nmsg: ", msg)
            # Extract parameter groups
            pred_params, classifier_params, bias_params = get_model_params_grouped(self.model, self.args)

            # Freeze classification and prediction parameters
            for param in classifier_params + pred_params:
                param.requires_grad = False
        
        self.num_params = count_parameters(self.model)
        
        self.optimizer = optimizer
        self.train_loader = dataloader['train']
        self.val_loader = dataloader['val']
        self.test_loader = dataloader['test']
        self.scaler = dataloader['scaler']
        self.load_path_baseline_used = False
        

        self.train_per_epoch = len(self.train_loader)
        if self.val_loader != None:
            self.val_per_epoch = len(self.val_loader)
        
        # log
        args.log_dir = get_log_dir(args)
        if os.path.isdir(args.log_dir) == False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.log_dir, debug=args.debug)
        self.logger.info('\nModel has {} M trainable parameters'.format(self.num_params/(1e6)))
        self.best_path = os.path.join(self.args.log_dir, 'best_model.pth')
        self.logs_dir = self.args.log_dir
        
        # create a panda object to log loss and acc
        self.training_stats = PD_Stats(
            os.path.join(args.log_dir, 'stats.pkl'), 
            ['epoch', 'train_loss', 'val_loss'],
        )
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))
        self.logger.info('Experiment configs are: {}'.format(args))
        self.logger.info('\nModel has {} M trainable parameters'.format(self.num_params/(1e6)))

        ema = [0.996, 1.0]
        ipe = args.ipe
        ipe_scale = 1.0
        num_epochs=args.num_epochs
        num_graphs = 8
        self.momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale*num_graphs)
                            for i in range(int(ipe*num_epochs*ipe_scale*num_graphs)+1))
    
    def _get_dummy_input(self, dataloader, args):
        # Construct a suitable dummy input based on your dataloader and args
        # Example:
        for batch in dataloader: 
            # print("batch.shape: ", batch[0].shape, batch[1].shape)  # batch.shape:  torch.Size([32, 35, 200, 2]) torch.Size([32, 1, 200, 2])
            if args.device == 'cuda':
                batch = batch[0].to('cuda')
            else:
                batch = batch[0].to('cpu')
                
            return batch
        
    def train_epoch(self, epoch, loss_weights, epoch_losses, epoch_losses_pred, epoch_losses_class, phase):
        self.model.train()
        
        total_loss = 0
        total_loss_pred = 0 
        total_loss_class = 0 
        max_train_batches = getattr(self.args, "max_train_batches", None)
        batches_seen = 0
        for batch_idx, (data, target, evs, _) in enumerate(self.train_loader):
            if max_train_batches is not None and batch_idx >= max_train_batches:
                break
            # print("data.shape: ", data.shape, target.shape)
            self.optimizer.zero_grad()
            
            # input shape: n,l,v,c; graph shape: v,v;
            repr1, repr1_cls = self.model(data, self.graph) # nvc
            

            loss, loss_pred, loss_class, loss_weights = self.model.loss(repr1, repr1_cls, evs, target, self.scaler, loss_weights, phase)
            # print("sep_loss: ", sep_loss)
            assert not torch.isnan(loss)
            loss.backward()

            
            # gradient clipping
            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    get_model_params([self.model]), 
                    self.args.max_grad_norm)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_loss_pred += loss_pred
            total_loss_class += loss_class
            batches_seen += 1
        
        if batches_seen == 0:
            raise ValueError("No training batches were processed.")

        train_epoch_loss = total_loss/batches_seen
        train_epoch_loss_pred = total_loss_pred/batches_seen
        train_epoch_loss_class = total_loss_class/batches_seen
        # Save losses for plotting
        epoch_losses.append(train_epoch_loss)
        epoch_losses_pred.append(train_epoch_loss_pred)
        epoch_losses_class.append(train_epoch_loss_class)
        self.logger.info(f'*******Train Epoch {epoch}: averaged Loss : {train_epoch_loss:.5f}, loss_pred: {train_epoch_loss_pred:.5f}, loss_class: {train_epoch_loss_class:.5f}')

        return train_epoch_loss, epoch_losses, epoch_losses_pred, epoch_losses_class, loss_weights
    
    def val_epoch(self, epoch, val_dataloader, loss_weights, phase):
        self.model.eval()
        
        total_val_loss = 0
        total_val_loss_pred = 0
        total_val_loss_class = 0
        evs_true = []
        evs_pred = []
        targets = []
        with torch.no_grad():
            max_eval_batches = getattr(self.args, "max_eval_batches", None)
            batches_seen = 0
            for batch_idx, (data, target, evs, _) in enumerate(val_dataloader):
                if max_eval_batches is not None and batch_idx >= max_eval_batches:
                    break
                repr1, repr1_cls = self.model(data, self.graph)
                loss, loss_pred, loss_class, _ = self.model.loss(repr1, repr1_cls, evs, target, self.scaler, loss_weights, phase, val=True)
                evs_true.append(evs)
                if not uses_ungated_bias_ablation(self.args):
                    evs_pred.append(self.model.classify_evs(repr1, repr1_cls))
                targets.append(self.scaler.inverse_transform(target))
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
                    total_val_loss_pred += loss_pred
                    total_val_loss_class += loss_class
                batches_seen += 1
        if batches_seen == 0:
            raise ValueError("No validation batches were processed.")
        evs_true = torch.cat(evs_true, dim=0).cpu()
        if evs_pred:
            evs_pred = torch.cat(evs_pred, dim=0).cpu()
        targets = torch.cat(targets, dim=0).cpu()
        val_loss = total_val_loss / batches_seen
        val_loss_pred = total_val_loss_pred / batches_seen
        val_loss_class = total_val_loss_class / batches_seen
        self.logger.info(f'*******Val Epoch {epoch}: averaged Loss : {val_loss:.5f}, loss_pred: {val_loss_pred:.5f}, loss_class: {val_loss_class:.5f}')
        # cm = plot_cm(evs_pred, evs_true, gt=None)
        # self.logger.info(f"Confusion Matrix: \n{cm}")
        return val_loss_pred, val_loss_class

    def save_weights(self, weights, epoch=None, directory="weight_data"):
        if epoch is not None:
            save_path = os.path.join(self.args.log_dir, f'learnable_weights_epoch_{epoch}.png')
        else:
            save_path = os.path.join(self.args.log_dir, f'learnable_weights.png')
        
        np.save(save_path, weights)
    
    def train_component(self, params_to_train, other_params, component_name, esp):
        if params_to_train is not None:
            for param in params_to_train:
                param.requires_grad = True
        if other_params is not None:
            for param in other_params:
                param.requires_grad = False

        train_epoch_losses = []
        val_epoch_losses = []
        train_epoch_losses_pred = []
        train_epoch_losses_class = []
        weight_history = []
        best_loss = float('inf')
        best_epoch = 0
        not_improved_count = 0
        start_time = time.time()
        current_weights = self.model.weights.detach().cpu().numpy()
        weight_history.append(current_weights)
        key_pressed = False
        def end_training():
            nonlocal key_pressed
            key_pressed = True
            print("Ctrl+Shift+K pressed. Ending training...")

        try:
            import keyboard
            keyboard.add_hotkey('ctrl+shift+k', end_training)
        except ImportError:
            self.logger.info("Optional keyboard hotkey support is unavailable; continuing without it.")
        cls_w = 1
        loss_weights = np.array([1, cls_w])

        for epoch in range(1, self.args.epochs + 1):
            if key_pressed:
                self.logger.info('Key press detected. Exiting training loop...')
                break

            self.logger.info('loss weights: {}'.format(loss_weights))
            use_loaded_baseline = (
                epoch == 1
                and self.args.load_path is not None
                and not self.load_path_baseline_used
                and component_name == getattr(self.args, "start_phase", "pred")
            )
            if use_loaded_baseline:
                self.logger.info('validating pretrained model')
                val_dataloader = self.val_loader if self.val_loader != None else self.test_loader
                val_loss_pred, val_loss_cls = self.val_epoch(epoch, val_dataloader, loss_weights, component_name)       
                val_epoch_loss = val_loss_cls if component_name == 'cls' else val_loss_pred
                val_epoch_losses.append(val_epoch_loss)
                best_loss = val_epoch_loss  
                self.best_path = self.args.load_path
                self.load_path_baseline_used = True

            train_epoch_loss, train_epoch_losses, train_epoch_losses_pred, train_epoch_losses_class, loss_weights = self.train_epoch(epoch, loss_weights, train_epoch_losses, train_epoch_losses_pred, train_epoch_losses_class, component_name)
            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break

            current_weights = self.model.weights.detach().cpu().numpy()
            weight_history.append(current_weights)

            if (epoch + 1) % 1 == 0 or epoch == self.args.epochs or epoch == 1:
                self.save_weights(np.array(weight_history))

            val_dataloader = self.val_loader if self.val_loader != None else self.test_loader
            val_loss_pred, val_loss_cls = self.val_epoch(epoch, val_dataloader, loss_weights, component_name)       
            val_epoch_loss = val_loss_cls if component_name == 'cls' else val_loss_pred     
            val_epoch_losses.append(val_epoch_loss)
            if not self.args.debug:
                self.training_stats.update((epoch, train_epoch_loss, val_epoch_loss))

            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                best_epoch = epoch
                not_improved_count = 0
                save_dict = {
                    "epoch": epoch, 
                    "model": self.model.state_dict(), 
                    "optimizer": self.optimizer.state_dict(),
                }
                if not self.args.debug:
                    # edit self.best_path to have component_name
                    self.best_path = os.path.join(self.args.log_dir, f'best_model_{component_name}.pth')
                    self.logger.info('**************Current best model saved to {}'.format(self.best_path))
                    
                    torch.save(save_dict, self.best_path)
            else:
                not_improved_count += 1

            if self.args.early_stop and not_improved_count == esp:
                self.logger.info(f"\n\n\nValidation performance didn\'t improve for {esp} epochs. Ending training for {component_name}.")
                self.logger.info("testing")
                break
        training_time = time.time() - start_time
        self.logger.info("== Training finished.\n"
                    "Total training time: {:.2f} min\t"
                    "best loss: {:.4f}\t"
                    "best epoch: {}\t".format(
                        (training_time / 60), 
                        best_loss, 
                        best_epoch))
        state_dict = save_dict if self.args.debug else torch.load(self.best_path, map_location=torch.device(self.args.device))
        self.model.load_state_dict(state_dict['model'])
        self.logger.info("== Test results.")
        evaluation_phase = (
            getattr(self.args, "evaluation_phase", None)
            or getattr(self.args, "stop_after_phase", "bias")
        )
        capture_predictions = (
            getattr(self.args, "capture_predictions", False)
            and component_name == evaluation_phase
        )
        test_results = self.test(
            self.model, self.test_loader, self.scaler, self.graph, self.logger,
            self.args, component_name, capture_predictions=capture_predictions,
        )
        results = {
            'best_val_loss': best_loss, 
            'best_val_epoch': best_epoch, 
            'test_results': test_results,
        }
        self.plot_losses(train_epoch_losses, val_epoch_losses, train_epoch_losses_pred, train_epoch_losses_class, component_name)
        
        return results


    def train(self):
        import shutil
        import os
        import time
        import numpy as np
        import torch

        

        current_directory = os.path.dirname(os.path.abspath(__file__))
        models_file_path = os.path.join(current_directory, 'models.py')
        layers_file_path = os.path.join(current_directory, 'layers.py')
        trainer_file_path = os.path.join(current_directory, 'trainer.py')
        main_file_path = os.path.join(os.path.dirname(current_directory), 'main.py')
        save_dir = self.logs_dir
        shutil.copy(models_file_path, save_dir)
        shutil.copy(layers_file_path, save_dir)
        shutil.copy(trainer_file_path, save_dir)
        shutil.copy(main_file_path, save_dir)
        self.logger.info('Model code files saved in: {}'.format(save_dir))

        cls_w = 1
        loss_weights = np.array([1, cls_w])
        epoch=1
        component_name = 'pred'
        self.logger.info('validating pretrained model')
        val_dataloader = self.val_loader if self.val_loader != None else self.test_loader
        val_loss_pred, val_loss_cls = self.val_epoch(epoch, val_dataloader, loss_weights, component_name)       
        self.logger.info("testing")
        test_results = self.test(self.model, self.test_loader, self.scaler, self.graph, self.logger, self.args, component_name)
        pred_params, classifier_params, bias_params = get_model_params_grouped(self.model, self.args)

        
        ## phase wise training. Load the saved model after every phase so we use the best model (best val loss) and not the latest model.
        start_phase = getattr(self.args, "start_phase", "pred")
        stop_after_phase = getattr(self.args, "stop_after_phase", "pred_2")
        start_idx = phase_index(start_phase)
        stop_idx = phase_index(stop_after_phase)
        if start_idx > stop_idx:
            raise ValueError(f"start_phase {start_phase} is after stop_after_phase {stop_after_phase}")

        # Phase-1 training:
        results = None
        if start_idx <= 0 <= stop_idx:
            phase1_train_params = pred_params
            phase1_other_params = bias_params + classifier_params
            if (
                uses_expanded_base_additive_ablation(self.args)
                or getattr(self.args, "ablation_mode", "original")
                == "event_weighted_ungated_end_to_end"
            ):
                self.logger.info(
                    "Training the prediction path and residual head together "
                    "in the prediction phase."
                )
                phase1_train_params = pred_params + bias_params
                phase1_other_params = classifier_params
            results = self.train_component(
                phase1_train_params, phase1_other_params, 'pred', esp=30)
            load_from = self.best_path
            if load_from is not None:
                state_dict = torch.load(
                    load_from, map_location=torch.device(self.args.device))
                msg = self.model.load_state_dict(state_dict['model'])
                print("loading pretrained model from: ", load_from)
                print("\nmsg: ", msg)
                # Extract parameter groups
                pred_params, classifier_params, bias_params = get_model_params_grouped(self.model, self.args)
        
        # Phase-2 training:
        if uses_ungated_bias_ablation(self.args):
            self.logger.info(
                "Ablation ungated_bias: skipping classifier phase and BCE loss; "
                "bias correction is added at every point."
            )
        elif start_idx <= 1 <= stop_idx:
            results = self.train_component(
                classifier_params, pred_params+bias_params, 'cls', esp=10)
            load_from = self.best_path
            if load_from is not None:
                state_dict = torch.load(
                    load_from, map_location=torch.device(self.args.device))
                msg = self.model.load_state_dict(state_dict['model'])
                print("loading pretrained model from: ", load_from)
                print("\nmsg: ", msg)
                # Extract parameter groups
                pred_params, classifier_params, bias_params = get_model_params_grouped(self.model, self.args)
        
        # Phase-3 training:
        phase3_mode = getattr(self.args, "phase3_mode", "original")
        if start_idx <= 2 <= stop_idx:
            if uses_ungated_bias_ablation(self.args):
                results = self.train_component(
                    pred_params+bias_params, classifier_params, 'bias', esp=30)
            elif phase3_mode == "original":
                results = self.train_component(
                    pred_params+bias_params, classifier_params, 'bias', esp=30)
            elif phase3_mode == "joint_separated":
                self.logger.info(
                    "Phase-3 joint_separated: classifier fine-tunes on BCE while "
                    "encoder/prediction/bias train on MAE with detached classifier gate."
                )
                results = self.train_component(
                    pred_params+bias_params+classifier_params, None, 'bias', esp=30)
            else:
                raise ValueError(f"Unsupported phase3_mode: {phase3_mode}")

            load_from = self.best_path
            if load_from is not None:
                state_dict = torch.load(
                    load_from, map_location=torch.device(self.args.device))
                msg = self.model.load_state_dict(state_dict['model'])
                print("loading pretrained model from: ", load_from)
                print("\nmsg: ", msg)
                # Extract parameter groups
                pred_params, classifier_params, bias_params = get_model_params_grouped(self.model, self.args)

        
        # Phase-4 training
        if start_idx <= 3 <= stop_idx:
            results = self.train_component(
                bias_params, classifier_params + pred_params, 'pred_2', esp=30)
        
        return results

    def plot_losses(self, train_epoch_losses, val_epoch_losses, train_epoch_losses_pred, train_epoch_losses_class, component_name):
            plt.figure(figsize=(12, 8))
            plt.plot(train_epoch_losses, label='Train Loss')
            plt.plot(val_epoch_losses, label='Val Loss (pred only)')
            labels = ["pred", "class"]
            plt.plot(train_epoch_losses_pred, label=f'Loss {labels[0]}')
            plt.plot(train_epoch_losses_class, label=f'Loss {labels[1]}')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Losses')
            plt.legend()
            plt.savefig(os.path.join(self.args.log_dir, f'losses_{component_name}.png'))

    @staticmethod
    def test(model, dataloader, scaler, graph, logger, args, phase, capture_predictions=False):
        model.eval()
        y_pred = []
        y_true = []
        evs_true = []
        with torch.no_grad():
            max_eval_batches = getattr(args, "max_eval_batches", None)
            for batch_idx, (data, target, evs, _) in enumerate(dataloader):
                if max_eval_batches is not None and batch_idx >= max_eval_batches:
                    break
                repr1, repr1_cls = model(data, graph)                
                pred_output = model.predict(repr1, repr1_cls, phase)
                y_true.append(target)
                y_pred.append(pred_output)
                evs_true.append(evs)
        if not y_true:
            raise ValueError("No test batches were processed.")
        y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))
        y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))
        # y_pred = torch.cat(y_pred, dim=0)
        evs_true = torch.cat(evs_true, dim=0).cpu()

        test_results = []
        detailed_results, artifacts = build_evaluation_details(
            y_pred, y_true, evs_true, capture_predictions=capture_predictions
        )
        flow_names = ("INFLOW", "OUTFLOW")
        for flow_idx in range(y_pred.shape[-1]):
            mae, eee = test_metrics(
                y_pred[..., flow_idx].cpu(),
                y_true[..., flow_idx].cpu(),
                evs=evs_true[..., flow_idx],
            )
            logger.info("{}, MAE: {:.2f}, EEE: {:.4f}".format(flow_names[flow_idx], mae, eee))
            test_results.append([mae, eee])
        model.last_test_details = detailed_results
        model.last_test_artifacts = artifacts
        return np.stack(test_results, axis=0)


def plot_cm(pred, true, gt=None):
    # Example data, replace these with your actual data
    # print("gt.shape: ", gt.shape, "pred.shape: ", pred.shape)
        
    # gt=None
    if gt is not None:
        mask_value = 5.0
        # gt = gt.cpu().numpy()
        mask = torch.gt(gt, mask_value).cpu()
        # print("==>", torch.sum(mask))
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    pred = pred.cpu().numpy()
    true = true.cpu().numpy()
    evs_pred_binary = (pred >= 0.5).astype(int)       # Threshold predictions at 0.2

    # Flatten the arrays
    evs_true_flat = true.flatten()
    evs_pred_flat = evs_pred_binary.flatten()

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(evs_true_flat, evs_pred_flat)
    return conf_matrix

