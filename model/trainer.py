import os
import time
import numpy as np
import torch
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

class Trainer(object):
    def __init__(self, model, optimizer, dataloader, graph, args):
        super(Trainer, self).__init__()
        self.model = model 
        self.graph = graph
        self.args = args

        if isinstance(self.model, torch.nn.Module):  # Check if it's a PyTorch module
            with torch.no_grad():  # Temporarily disable gradient calculations
                dummy_input = self._get_dummy_input(dataloader['val'], args)  
                _ = self.model(dummy_input, self.graph)  # Trigger initialization

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
        num_graphs = 17
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
   
    def train_epoch(self, epoch, loss_weights, epoch_losses, sep_epoch_losses):
        self.model.train()
        
        total_loss = 0
        total_sep_loss = np.zeros(1) 
        for batch_idx, (data, target) in enumerate(self.train_loader):
            # print("data.shape: ", data.shape, target.shape)
            self.optimizer.zero_grad()
            
            # input shape: n,l,v,c; graph shape: v,v;
            repr1, learnable_graph, z, h = self.model(data, self.graph) # nvc
            

            loss, sep_loss = self.model.loss(repr1, learnable_graph, target, self.scaler, loss_weights, z, h)
            # print("sep_loss: ", sep_loss)
            assert not torch.isnan(loss)
            loss.backward()

            
            # gradient clipping
            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    get_model_params([self.model]), 
                    self.args.max_grad_norm)
            self.optimizer.step()
            with torch.no_grad():
                m = next(self.momentum_scheduler)
                for param_q, param_k in zip(self.model.encoder.parameters(), self.model.target_encoder.parameters()):
                    param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

            total_loss += loss.item()
            total_sep_loss += sep_loss
        if epoch % 10 == 0:
            plot = "no"
            if plot == "image":
                import networkx as nx
                import matplotlib.pyplot as plt

                if torch.is_tensor(learnable_graph):
                    graph_matrix = learnable_graph.detach().cpu().numpy()  # Convert to NumPy
                else:
                    graph_matrix = learnable_graph  # Assuming it's already a NumPy array

                G = nx.Graph(graph_matrix)  # Create the NetworkX graph
                nx.draw(G, with_labels=True) 
                save_path = os.path.join(self.args.log_dir, f'learnable_graph_epoch_{epoch}.png')
                plt.savefig(save_path)
                plt.close()  

            elif plot == "matrix":
                import seaborn as sns
                import matplotlib.pyplot as plt
                adj = learnable_graph.detach().cpu().numpy()
                plt.figure(figsize=(20, 20))
                sns.heatmap(adj, cmap='viridis', annot=True)  # annot=True shows values
                plt.title('Adjacency Matrix Visualization')
                save_path = os.path.join(self.args.log_dir, f'learnable_graph_epoch_{epoch}.png')
                plt.savefig(save_path)

            elif plot == "npArray":
                # print("==>", learnable_graph.detach().cpu().numpy().shape)
                np.save(os.path.join(self.args.log_dir, f'learnable_graph_epoch_{epoch}.npy'), learnable_graph.detach().cpu().numpy())

        train_epoch_loss = total_loss/self.train_per_epoch
        total_sep_loss = total_sep_loss/self.train_per_epoch
        # Save losses for plotting
        epoch_losses.append(train_epoch_loss)
        sep_epoch_losses.append(total_sep_loss)
        self.logger.info('*******Train Epoch {}: averaged Loss : {:.6f}'.format(epoch, train_epoch_loss))

        return train_epoch_loss, total_sep_loss, epoch_losses, sep_epoch_losses
    
    def val_epoch(self, epoch, val_dataloader, loss_weights):
        self.model.eval()
        
        total_val_loss = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_dataloader):
                repr1, repr2, z, h = self.model(data, self.graph)
                loss, sep_loss = self.model.loss(repr1, repr2, target, self.scaler, loss_weights, z, h)

                if not torch.isnan(loss):
                    total_val_loss += loss.item()
        val_loss = total_val_loss / len(val_dataloader)
        self.logger.info('*******Val Epoch {}: averaged Loss : {:.6f}'.format(epoch, val_loss))
        return val_loss

    def save_weights(self, weights, epoch, directory="weight_data"):
        save_path = os.path.join(self.args.log_dir, f'learnable_weights_epoch_{epoch}.png')
        np.save(save_path, weights)
    
    def train(self):
        train_epoch_losses = []
        val_epoch_losses = []
        sep_epoch_losses = []
        weight_history = []
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
            self.logger.info('loss weights: {}'.format(loss_weights))
            train_epoch_loss, loss_t, train_epoch_losses, sep_epoch_losses = self.train_epoch(epoch, loss_weights, train_epoch_losses, sep_epoch_losses)
            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break

            current_weights = self.model.weights.detach().cpu().numpy()
            weight_history.append(current_weights)

            # Save weights every 5 epochs, for example
            if (epoch + 1) % 5 == 0 or epoch == self.args.epochs:  # Also save on the last epoch
                self.save_weights(np.array(weight_history), epoch + 1)
            
            val_dataloader = self.val_loader if self.val_loader != None else self.test_loader
            val_epoch_loss = self.val_epoch(epoch, val_dataloader, loss_weights)       
            val_epoch_losses.append(val_epoch_loss)
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
        
        

        def plot_losses(train_epoch_losses, val_epoch_losses, sep_epoch_losses):
            plt.figure(figsize=(12, 8))

            # Plotting the total loss
            plt.plot(train_epoch_losses, label='Train Loss')
            plt.plot(val_epoch_losses, label='Val Loss')

            labels = ["i-jepa"]
            # Plotting the separate losses
            print("sep_epoch_losses: ", sep_epoch_losses)
            plt.plot(sep_epoch_losses, label=f'Loss {labels[0]}')

            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Losses')
            plt.legend()
            # plt.show()
            plt.savefig(os.path.join(self.args.log_dir, 'losses.png'))


        # test
        state_dict = save_dict if self.args.debug else torch.load(
            self.best_path, map_location=torch.device(self.args.device))
        self.model.load_state_dict(state_dict['model'])
        self.logger.info("== Test results.")
        test_results = self.test(self.model, self.test_loader, self.scaler, 
                                self.graph, self.logger, self.args)
        results = {
            'best_val_loss': best_loss, 
            'best_val_epoch': best_epoch, 
            'test_results': test_results,
        }

        plot_losses(train_epoch_losses, val_epoch_losses, sep_epoch_losses)
        return results

    @staticmethod
    def test(model, dataloader, scaler, graph, logger, args):
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dataloader):
                repr1, repr2, _, _ = model(data, graph)                
                pred_output = model.predict(repr1, repr2)

                y_true.append(target)
                y_pred.append(pred_output)
        y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))
        y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))

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

        return np.stack(test_results, axis=0)



        

