import os
import time
import numpy as np
import torch
# torch.autograd.set_detect_anomaly(True)

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

class Trainer(object):
    def __init__(self, model, optimizer, dataloader, graph, args):
        super(Trainer, self).__init__()
        self.model = model 
        self.optimizer = optimizer
        self.train_loader = dataloader['train']
        self.val_loader = dataloader['val']
        self.test_loader = dataloader['test']
        self.scaler = dataloader['scaler']
        self.graph = graph
        self.args = args

        self.train_per_epoch = len(self.train_loader)
        if self.val_loader != None:
            self.val_per_epoch = len(self.val_loader)
        
        # log
        args.log_dir = get_log_dir(args)
        if os.path.isdir(args.log_dir) == False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.log_dir, debug=args.debug)
        self.best_path = os.path.join(self.args.log_dir, 'best_model.pth')
        
        # create a panda object to log loss and acc
        self.training_stats = PD_Stats(
            os.path.join(args.log_dir, 'stats.pkl'), 
            ['epoch', 'train_loss', 'val_loss'],
        )
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))
        self.logger.info('Experiment configs are: {}'.format(args))
    
    def train_epoch(self, epoch, loss_weights):
        self.model.train()
        
        total_loss = 0
        total_sep_loss = np.zeros(3) 
        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            
            # input shape: n,l,v,c; graph shape: v,v;
            repr1, repr2 = self.model(data, self.graph) # nvc
            loss, sep_loss = self.model.loss(repr1, repr2, target, self.scaler, loss_weights)
            assert not torch.isnan(loss)
            loss.backward()

            # gradient clipping
            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    get_model_params([self.model]), 
                    self.args.max_grad_norm)
            self.optimizer.step()
            total_loss += loss.item()
            total_sep_loss += sep_loss

        train_epoch_loss = total_loss/self.train_per_epoch
        total_sep_loss = total_sep_loss/self.train_per_epoch
        self.logger.info('*******Train Epoch {}: averaged Loss : {:.6f}'.format(epoch, train_epoch_loss))

        return train_epoch_loss, total_sep_loss
    
    def val_epoch(self, epoch, val_dataloader, loss_weights):
        self.model.eval()
        
        total_val_loss = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_dataloader):
                repr1, repr2 = self.model(data, self.graph)
                loss, sep_loss = self.model.loss(repr1, repr2, target, self.scaler, loss_weights)

                if not torch.isnan(loss):
                    total_val_loss += loss.item()
        val_loss = total_val_loss / len(val_dataloader)
        self.logger.info('*******Val Epoch {}: averaged Loss : {:.6f}'.format(epoch, val_loss))
        return val_loss

    def train(self):
        best_loss = float('inf')
        best_epoch = 0
        not_improved_count = 0
        start_time = time.time()

        loss_tm1 = loss_t = np.ones(3) #(1.0, 1.0, 1.0)
        for epoch in range(1, self.args.epochs + 1):
            # dwa mechanism to balance optimization speed for different tasks
            if self.args.use_dwa:
                loss_tm2 = loss_tm1
                loss_tm1 = loss_t
                if (epoch == 1) or (epoch == 2):
                    loss_weights = dwa(loss_tm1, loss_tm1, self.args.temp)
                else:
                    loss_weights  = dwa(loss_tm1, loss_tm2, self.args.temp)
            # self.logger.info('loss weights: {}'.format(loss_weights))
            train_epoch_loss, loss_t = self.train_epoch(epoch, loss_weights)
            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break
            
            val_dataloader = self.val_loader if self.val_loader != None else self.test_loader
            val_epoch_loss = self.val_epoch(epoch, val_dataloader, loss_weights)       
            if not self.args.debug:
                self.training_stats.update((epoch, train_epoch_loss, val_epoch_loss))

            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                best_epoch = epoch
                not_improved_count = 0
                # save the best state
                save_dict = {
                    "epoch": epoch, 
                    "model": self.model.state_dict(), 
                    "optimizer": self.optimizer.state_dict(),
                }
                if not self.args.debug:
                    self.logger.info('**************Current best model saved to {}'.format(self.best_path))
                    torch.save(save_dict, self.best_path)
            else:
                not_improved_count += 1

            # early stopping
            if self.args.early_stop and not_improved_count == self.args.early_stop_patience:
                self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                "Training stops.".format(self.args.early_stop_patience))
                break   

        training_time = time.time() - start_time
        self.logger.info("== Training finished.\n"
                    "Total training time: {:.2f} min\t"
                    "best loss: {:.4f}\t"
                    "best epoch: {}\t".format(
                        (training_time / 60), 
                        best_loss, 
                        best_epoch))
        
        # test
        state_dict = save_dict if self.args.debug else torch.load(
            self.best_path, map_location=torch.device(self.args.device))
        self.model.load_state_dict(state_dict['model'])
        self.logger.info("== Test results.")
        test_results = self.test(self.model, self.test_loader, self.scaler, 
                                self.graph, self.logger, self.args)
        test_rolling_results = self.test_rolling(self.model, self.test_loader, self.scaler, 
                                self.graph, self.logger, self.args)
        results = {
            'best_val_loss': best_loss, 
            'best_val_epoch': best_epoch, 
            'test_results': test_results,
            'test_rolling_results': test_rolling_results,
        }
        return results

    @staticmethod
    def test(model, dataloader, scaler, graph, logger, args):
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dataloader):
                repr1, repr2 = model(data, graph)                
                pred_output = model.predict(repr1, repr2)

                y_true.append(target)
                y_pred.append(pred_output)
        y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))
        y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))

        test_results = []
        # inflow
        mae, mape = test_metrics(y_pred[..., 0], y_true[..., 0])
        logger.info("test, MAE: {:.2f}, test MAPE: {:.4f}%".format(mae, mape*100))
        test_results.append([mae, mape])
        # outflow 
        # mae, mape = test_metrics(y_pred[..., 1], y_true[..., 1])
        # logger.info("OUTFLOW, MAE: {:.2f}, MAPE: {:.4f}%".format(mae, mape*100))
        # test_results.append([mae, mape]) 

        return np.stack(test_results, axis=0)
    
    # test_rolling
    @staticmethod
    def test_rolling(model, dataloader, scaler, graph, logger, args):
        model.eval()
        y_pred = []
        y_true = []
        prediction_buffer = None  # Buffer to hold predictions for future time steps

        # torch.Size([batch_size=1, time_steps=19, no.ofnodes=285, features=1]) data shape
        # torch.Size([1, 1, 285, 1]) pred_output shape
        # print("len(dataloader): ", len(dataloader))
        with torch.no_grad():
            for idx, (data, target) in enumerate(dataloader):
                # print(f"idx: {idx}, data.shape: {data.shape}, target.shape: {target.shape}")
                # Update data with predictions from buffer
                if prediction_buffer is not None:
                    for t in range(len(prediction_buffer)):
                        if t <= data.shape[1]:  # Ensure the time step exists in the data
                            # print("data.shape: ", data.shape, "data[:, t, :, :].shape: ", data[:, t, :, :].shape, "prediction_buffer[t].shape: ", prediction_buffer[t].shape)
                            # print(f"for data at index {idx} replacing data[:, {t}, :, :] with prediction_buffer[{t}]")
                            data[:, t, :, :] = prediction_buffer[t].squeeze(1)

                repr1, repr2 = model(data, graph)
                pred_output = model.predict(repr1, repr2)

                y_true.append(target)
                y_pred.append(pred_output)

                # Update the prediction buffer
                if prediction_buffer is None:
                    prediction_buffer = [pred_output]
                else:
                    prediction_buffer.insert(0, pred_output)  ## insert at 0th index
                    prediction_buffer = prediction_buffer[:data.shape[1]]  # Keep buffer size same as the number of time steps

        y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))
        y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))

        test_results = []
        # inflow
        mae, mape = test_metrics(y_pred[..., 0], y_true[..., 0])
        logger.info("rolling, MAE: {:.2f}, rolling MAPE: {:.4f}%".format(mae, mape*100))
        test_results.append([mae, mape])

        return np.stack(test_results, axis=0)




        

