import warnings 
warnings.filterwarnings('ignore')

import sys
sys.path.append('.')
sys.path.append('..')
import yaml 
import argparse
import traceback
import time
import torch
import json

from model.trainer import Trainer
from model.parameter_groups import format_param_group_counts, get_model_params_grouped
from lib.dataloader import get_dataloader
from lib.utils import (
    init_seed,
    # get_model_params,
    load_graph, 
)
import os


def _to_jsonable(value):
    if isinstance(value, argparse.Namespace):
        return {key: _to_jsonable(val) for key, val in vars(value).items()}
    if isinstance(value, dict):
        return {str(key): _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except TypeError:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _build_results_payload(args, results):
    payload = {
        "dataset": args.dataset,
        "mode": args.mode,
        "seed": args.seed,
        "comment": args.comment,
        "training_recipe": getattr(args, "training_recipe", "full"),
        "experiment_dir": getattr(args, "log_dir", None),
        "base_checkpoint_path": args.load_path,
        "hyperparameters": {
            "lr_init": args.lr_init,
            "epochs": args.epochs,
            "base_epochs": getattr(args, "base_epochs", None),
            "base_early_stop_patience": getattr(args, "base_early_stop_patience", None),
            "tail_stage1_epochs": getattr(args, "tail_stage1_epochs", None),
            "tail_stage1_early_stop_patience": getattr(args, "tail_stage1_early_stop_patience", None),
            "early_stop": args.early_stop,
            "early_stop_patience": args.early_stop_patience,
            "training_recipe": getattr(args, "training_recipe", "full"),
            "input_length": getattr(args, "input_length", None),
            "output_length": getattr(args, "output_length", None),
            "tail_threshold_q": getattr(args, "tail_threshold_q", None),
            "tail_lambda_cls": getattr(args, "tail_lambda_cls", None),
            "tail_lambda_gpd": getattr(args, "tail_lambda_gpd", None),
            "tail_mae_weight": getattr(args, "tail_mae_weight", None),
            "tail_phase_selection_metric": getattr(args, "tail_phase_selection_metric", None),
            "tail_schedule": getattr(args, "tail_schedule", "static"),
            "tail_magnitude_mode": getattr(args, "tail_magnitude_mode", "gpd"),
            "tail_classifier_loss_type": getattr(args, "tail_classifier_loss_type", "bce"),
            "tail_pos_weight_multiplier": getattr(args, "tail_pos_weight_multiplier", None),
            "tail_focal_gamma": getattr(args, "tail_focal_gamma", None),
            "tail_focal_alpha_pos": getattr(args, "tail_focal_alpha_pos", None),
            "tail_xi_min": getattr(args, "tail_xi_min", None),
            "tail_xi_max": getattr(args, "tail_xi_max", None),
            "tail_eps": getattr(args, "tail_eps", None),
            "joint_refine_epochs": getattr(args, "joint_refine_epochs", None),
            "joint_refine_early_stop_patience": getattr(args, "joint_refine_early_stop_patience", None),
            "joint_refine_pred_lr_scale": getattr(args, "joint_refine_pred_lr_scale", None),
            "joint_refine_classifier_lr_scale": getattr(args, "joint_refine_classifier_lr_scale", None),
            "joint_refine_gpd_lr_scale": getattr(args, "joint_refine_gpd_lr_scale", None),
            "joint_refine_lambda_cls": getattr(args, "joint_refine_lambda_cls", None),
            "joint_refine_lambda_gpd": getattr(args, "joint_refine_lambda_gpd", None),
            "joint_refine_lambda_mae": getattr(args, "joint_refine_lambda_mae", None),
            "joint_refine_recompute_tail_u": getattr(args, "joint_refine_recompute_tail_u", None),
        },
        "results": _to_jsonable(results),
    }
    if isinstance(results, dict):
        if "pred" in results and results["pred"] is not None:
            payload["pred"] = _to_jsonable(results["pred"])
        if "tail" in results and results["tail"] is not None:
            payload["tail"] = _to_jsonable(results["tail"])
    elif args.mode == "test" and results is not None:
        phase = "pred" if getattr(args, "training_recipe", "full") == "pred_only" else "tail"
        payload[phase] = {
            "test_metrics": _to_jsonable(Trainer.format_test_results(results)),
        }
    return payload


def _write_results_file(args, results):
    if not hasattr(args, "log_dir") or args.log_dir is None or results is None:
        return
    os.makedirs(args.log_dir, exist_ok=True)
    results_path = os.path.join(args.log_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as results_file:
        json.dump(_build_results_payload(args, results), results_file, indent=2)

def model_supervisor(args):
    init_seed(args.seed)
    if not torch.cuda.is_available():
        args.device = 'cpu'

    if args.dataset != 'PEMS04' and (not hasattr(args, 'evs_key') or not args.evs_key):
        raise KeyError("Config is missing required 'evs_key'. Add it to the dataset YAML.")
    
    # if args.load_path is None:
    from model.models import STSSL
    # else:
    #     model_dir = os.path.dirname(args.load_path)
    #     sys.path.append(model_dir)
    #     from models import STSSL

    ## load dataset
    dataloader = get_dataloader(
        data_dir=args.data_dir, 
        dataset=args.dataset, 
        batch_size=args.batch_size, 
        test_batch_size=args.test_batch_size,
        evs_key=args.evs_key,
        scalar_type='Standard',
        pems04_evs_quantile=float(getattr(args, 'pems04_evs_quantile', 0.95)),
    )
    graph = load_graph(args.graph_file, device=args.device)
    args.num_nodes = len(graph)
    
    args.ipe = len(dataloader['train'])
    ## init model and set optimizer
    model = STSSL(args).to(args.device)
    print(format_param_group_counts(model))

    pred_params, classifier_params, gpd_params = get_model_params_grouped(model)
    optimizer = torch.optim.Adam([
        {"params":pred_params, 
        "lr":args.lr_init, 
        "eps":1.0e-8, 
        'weight_decay':0, 
        "amsgrad":False},
        
        {"params":classifier_params, 
        "lr":args.lr_init, 
        "eps":1.0e-8, 
        'weight_decay':0, 
        "amsgrad":True},

        {"params":gpd_params, 
        "lr":args.lr_init, 
        "eps":1.0e-8, 
        'weight_decay':1.0e-8, 
        "amsgrad":True} 
    ])

    ## start training
    trainer = Trainer(
        model=model, 
        optimizer=optimizer, 
        dataloader=dataloader,
        graph=graph, 
        args=args
    )
    results = None
    try:
        if args.mode == 'train':
            results = trainer.train() # best_eval_loss, best_epoch
            print("Training done")
        elif args.mode == 'test':
            # test
            state_dict = torch.load(
                args.best_path,
                map_location=torch.device(args.device)
            )
            model.load_state_dict(state_dict['model'])
            print("Load saved model")
            test_phase = 'pred' if getattr(args, 'training_recipe', 'full') == 'pred_only' else 'tail'
            results = trainer.test(model, dataloader['test'], dataloader['scaler'],
                        graph, trainer.logger, trainer.args, test_phase)
        else:
            raise ValueError
    except:
        trainer.logger.info(traceback.format_exc())
    if results is not None:
        _write_results_file(trainer.args, results)
    return results

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', '-cf', default='configs/NYCTaxi.yaml', 
                    type=str, help='the configuration to use')

    parser.add_argument('--mode', default="train", type=str, help='train/test')
    parser.add_argument('--best_path', default="/path/to/trained/model", type=str, help='path to the best model to load for testing')
    parser.add_argument('--S_Loss', default=0, type=int, help='use S_Loss or not')
    parser.add_argument('--T_Loss', default=0, type=int, help='use T_Loss or not')
    parser.add_argument('--seed', "-s", default=1, type=int, help='random seed to use')
    parser.add_argument('--comment', "-c", default="noComment", type=str, help='comment about the experiment')
    parser.add_argument('--cheb_order', "-K", default=3, type=int, help='calculate the chebyshev polynomials up to this order')
    parser.add_argument('--graph_init', "-g", default="8_neighbours", type=str, help='how to initialize the learnable graph')
    
    """
    if you pass bool flags in cli it will automatically make it true, irrespective if you pass True or False. If you dont pass it then it uses the default value.
    """
    parser.add_argument('--self_attention_flag', "-sa", default=True, type=bool, help='wether to self attention before pred')
    parser.add_argument('--cross_attention_flag', "-ca", default=False, type=bool, help='wether to cross attention before pred')
    parser.add_argument('--feedforward_flag', "-ff", default=False, type=bool, help='wether to feedforward')
    parser.add_argument('--layer_norm_flag', "-ln", default=False, type=bool, help='wether to layernorm')
    parser.add_argument('--additional_sa_flag', "-asa", default=False, type=bool, help='wether to additional SA')
    parser.add_argument('--learnable_flag', "-lf", default=False, type=bool, help='wether to use learnable adj matrix')
    parser.add_argument('--rank', "-r", default=0, type=int, help='rank of adj matrix')
    parser.add_argument('--pos_emb_flag', "-pef", default=False, type=bool, help='wether to add pos_emb')
    parser.add_argument('--add_8', "-a8", default=False, type=bool, help='wether to add 8_neighbours')
    parser.add_argument('--add_eye', "-ai", default=False, type=bool, help='wether to add eye')
    parser.add_argument('--add_x_encoder', "-axe", default=False, type=bool, help='wether to add output of encoder')
    parser.add_argument('--freeze_encoder', "-fe", default=False, type=bool, help='wether to freeze encoder')
    parser.add_argument('--threshold_adj_mx', "-tadj", default=False, type=bool, help='wether to threshold the learnt adj_mx')
    parser.add_argument('--affinity_conv', "-afc", default=False, type=bool, help='wether to affinity conv')
    parser.add_argument('--loss', "-l", default="mae", type=str, help='mae/mse')
    parser.add_argument('--load_path', "-lp", default=None, type=str, help='path to load pretrained model from')
    parser.add_argument('--variant', "-v", default=None, type=str, help='which variant of model to use. pred/tail')

    # parser.add_argument('--input_length', default=0, type=int, help='# of samples to use for context')
    args = parser.parse_args()
    print(f'Starting experiment with configurations in {args.config_filename}...')
    
    time.sleep(3)
    configs = yaml.load(
        open(args.config_filename), 
        Loader=yaml.FullLoader
    )
    
    configs['S_Loss'] = args.S_Loss
    configs['T_Loss'] = args.T_Loss
    configs['seed'] = args.seed
    configs['comment'] = args.comment
    configs['cheb_order'] = args.cheb_order
    configs['graph_init'] = args.graph_init
    configs['self_attention_flag'] = args.self_attention_flag
    configs['cross_attention_flag'] = args.cross_attention_flag
    configs['feedforward_flag'] = args.feedforward_flag
    configs['layer_norm_flag'] = args.layer_norm_flag
    configs['additional_sa_flag'] = args.additional_sa_flag
    configs['learnable_flag'] = args.learnable_flag
    configs['pos_emb_flag'] = args.pos_emb_flag
    configs['rank'] = args.rank
    configs['add_8'] = args.add_8
    configs['add_eye'] = args.add_eye
    configs['add_x_encoder'] = args.add_x_encoder
    configs['freeze_encoder'] = args.freeze_encoder
    configs['threshold_adj_mx'] = args.threshold_adj_mx
    configs['affinity_conv'] = args.affinity_conv
    configs['loss'] = args.loss
    configs['load_path'] = args.load_path
    configs['variant'] = args.variant

    configs.setdefault('evs_key', None)
    configs.setdefault('pems04_evs_quantile', 0.95)
    configs.setdefault('output_length', 1)
    
    # configs['input_length'] = args.input_length
    # experimentName = "pred_" + str(args.input_length) + "_"
    experimentName = "pred_"
    if args.S_Loss == 1:
        experimentName += "+S"

    if args.T_Loss == 1:
        experimentName += "+T"
    experimentName += f"_seed={args.seed}"
    
    configs["experimentName"] = experimentName
    print(f'Starting experiment with configurations {configs}...')
    args = argparse.Namespace(**configs)
    model_supervisor(args)
