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
from lib.metrics import test_metrics

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_params_grouped(model):
    pred_params = []
    classifier_params = []
    bias_params = []
    for name, param in model.named_parameters():
        if 'cls' in name:
            classifier_params.append(param)
        elif "bias" in name:    
            bias_params.append(param)
        else:
            pred_params.append(param)
    return pred_params, classifier_params, bias_params

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
            pred_params, classifier_params, bias_params = get_model_params_grouped(self.model)

            # Freeze classification and prediction parameters
            for param in classifier_params + pred_params:
                param.requires_grad = False
        
        self.num_params = count_parameters(self.model)
        
        self.optimizer = optimizer
        self.train_loader = dataloader['train']
        self.val_loader = dataloader['val']
        self.test_loader = dataloader['test']
        self.scaler = dataloader['scaler']
        

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
                
            return batch
        
    def train_epoch(self, epoch, loss_weights, epoch_losses, epoch_losses_pred, epoch_losses_class, phase):
        self.model.train()
        
        total_loss = 0
        total_loss_pred = 0 
        total_loss_class = 0 
        for batch_idx, (data, target, evs, _) in enumerate(self.train_loader):
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
        
                
        train_epoch_loss = total_loss/self.train_per_epoch
        train_epoch_loss_pred = total_loss_pred/self.train_per_epoch
        train_epoch_loss_class = total_loss_class/self.train_per_epoch
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
            for batch_idx, (data, target, evs, _) in enumerate(val_dataloader):
                repr1, repr1_cls = self.model(data, self.graph)
                loss, loss_pred, loss_class, _ = self.model.loss(repr1, repr1_cls, evs, target, self.scaler, loss_weights, phase, val=True)
                evs_true.append(evs)
                evs_pred.append(self.model.classify_evs(repr1, repr1_cls))
                targets.append(self.scaler.inverse_transform(target))
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
                    total_val_loss_pred += loss_pred
                    total_val_loss_class += loss_class
        evs_true = torch.cat(evs_true, dim=0).cpu()
        evs_pred = torch.cat(evs_pred, dim=0).cpu()
        targets = torch.cat(targets, dim=0).cpu()
        val_loss = total_val_loss / len(val_dataloader)
        val_loss_pred = total_val_loss_pred / len(val_dataloader)
        val_loss_class = total_val_loss_class / len(val_dataloader)
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
        import keyboard
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

        keyboard.add_hotkey('ctrl+shift+k', end_training)
        cls_w = 1
        loss_weights = np.array([1, cls_w])

        for epoch in range(1, self.args.epochs + 1):
            if key_pressed:
                self.logger.info('Key press detected. Exiting training loop...')
                break

            self.logger.info('loss weights: {}'.format(loss_weights))
            if epoch == 1 and self.args.load_path is not None:
                self.logger.info('validating pretrained model')
                val_dataloader = self.val_loader if self.val_loader != None else self.test_loader
                val_loss_pred, val_loss_cls = self.val_epoch(epoch, val_dataloader, loss_weights, component_name)       
                val_epoch_loss = val_loss_cls if component_name == 'cls' else val_loss_pred
                val_epoch_losses.append(val_epoch_loss)
                best_loss = val_epoch_loss  
                self.best_path = self.args.load_path

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
        test_results = self.test(self.model, self.test_loader, self.scaler, self.graph, self.logger, self.args, component_name)
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
        pred_params, classifier_params, bias_params = get_model_params_grouped(self.model)

        
        if self.args.variant == "pred":
            results = self.train_component(
                pred_params, bias_params+classifier_params, 'pred', esp=30)
        elif self.args.variant == "cls":
            results = self.train_component(
                pred_params+classifier_params, bias_params, 'cls', esp=30)
        elif self.args.variant == "bias":
            results = self.train_component(
                pred_params+classifier_params+bias_params, None, 'bias', esp=30)
        

        # load_from = self.best_path
        # if load_from is not None:
        #     state_dict = torch.load(
        #         load_from, map_location=torch.device(self.args.device))
        #     msg = self.model.load_state_dict(state_dict['model']) 
        #     print("loading pretrained model from: ", load_from)
        #     print("\nmsg: ", msg)
        #     # Extract parameter groups
        #     pred_params, classifier_params, bias_params = get_model_params_grouped(self.model)

        
        # # Train the bias parameters until convergence
        # results = self.train_component(
        #     bias_params + classifier_params + pred_params, None, 'pred_2', esp=10)
        
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
    def test(model, dataloader, scaler, graph, logger, args, phase):
        model.eval()
        y_pred = []
        y_true = []
        evs_true = []
        evs_pred = []
        with torch.no_grad():
            for batch_idx, (data, target, evs, _) in enumerate(dataloader):
                repr1, repr1_cls = model(data, graph)                
                pred_output = model.predict(repr1, repr1_cls, phase)
                pred_evs = model.classify_evs(repr1, repr1_cls)
                y_true.append(target)
                y_pred.append(pred_output)
                evs_true.append(evs)
                evs_pred.append(pred_evs)
        y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))
        y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))
        # y_pred = torch.cat(y_pred, dim=0)
        evs_true = torch.cat(evs_true, dim=0).cpu()
        evs_pred = torch.cat(evs_pred, dim=0).cpu()

        test_results = []
        # inflow
        # print("y_pred.shape: ", y_pred.shape, "y_true.shape: ", y_true.shape)
        mae, mape = test_metrics(y_pred[..., 0], y_true[..., 0])
        logger.info("INFLOW, MAE: {:.2f}, MAPE: {:.4f}%".format(mae, mape*100))
        test_results.append([mae, mape])
        # outflow 
        mae, mape = test_metrics(y_pred[..., 1], y_true[..., 1])
        logger.info("OUTFLOW, MAE: {:.2f}, MAPE: {:.4f}%".format(mae, mape*100))
        test_results.append([mae, mape]) 
        cm = plot_cm(evs_pred, evs_true, gt=None)
        logger.info(f"Confusion Matrix: \n{cm}")
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

