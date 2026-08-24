import warnings 
warnings.filterwarnings('ignore')

import sys
sys.path.append('.')
sys.path.append('..')
import yaml 
import argparse
import traceback
import torch

from model.trainer import Trainer, get_model_params_grouped
from lib.dataloader import get_dataloader
from lib.utils import (
    init_seed,
    # get_model_params,
    load_graph, 
)


def resolve_device(requested_device=None):
    """Resolve a requested device without probing CUDA for explicit CPU use."""
    if requested_device is None or not str(requested_device).strip():
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    requested = str(requested_device)
    if requested.lower().startswith('cuda'):
        return requested if torch.cuda.is_available() else 'cpu'
    return requested


def model_supervisor(args):
    init_seed(args.seed)
    args.device = resolve_device(getattr(args, 'device', None))
    
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
        scalar_type='Standard',
        scaler_fit=getattr(args, "scaler_fit", "train_val"),
        event_percentile=getattr(args, "event_percentile", None),
        event_mask_protocol=getattr(args, "event_mask_protocol", "legacy_raw_p90_v1"),
        event_label_source=getattr(args, "event_label_source", "legacy_file_unverified"),
        device=args.device,
    )
    graph = load_graph(args.graph_file, device=args.device)
    args.num_nodes = len(graph)
    
    args.ipe = len(dataloader['train'])
    ## init model and set optimizer
    model = STSSL(args).to(args.device)
    
    pred_params, classifier_params, bias_params = get_model_params_grouped(model, args)
    parameter_counts = {
        "prediction": sum(p.numel() for p in pred_params),
        "classifier": sum(p.numel() for p in classifier_params),
        "residual": sum(p.numel() for p in bias_params),
    }
    if getattr(args, "stop_after_phase", "pred_2") == "pred":
        if getattr(args, "ablation_mode", "original") == "event_weighted_ungated_end_to_end":
            active_groups = ("prediction", "residual")
        else:
            active_groups = ("prediction",)
    elif getattr(args, "ablation_mode", "original") in {
        "ungated_bias",
        "event_weighted_ungated_residual",
        "event_weighted_dual_ungated",
    }:
        active_groups = ("prediction", "residual")
    else:
        active_groups = ("prediction", "classifier", "residual")
    parameter_counts["active"] = sum(parameter_counts[name] for name in active_groups)
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

        {"params":bias_params, 
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
            if results is not None:
                results["test_details"] = getattr(model, "last_test_details", None)
                results["parameter_counts"] = parameter_counts
                if getattr(args, "capture_predictions", False):
                    results["test_artifacts"] = getattr(model, "last_test_artifacts", None)
            print("Training done")
        elif args.mode == 'test':
            # test
            state_dict = torch.load(
                args.best_path,
                map_location=torch.device(args.device)
            )
            model.load_state_dict(state_dict['model'])
            print("Load saved model")
            test_results = trainer.test(
                model, dataloader['test'], dataloader['scaler'], graph,
                trainer.logger, trainer.args,
                getattr(args, "evaluation_phase", None)
                or getattr(args, "stop_after_phase", "bias"),
                capture_predictions=getattr(args, "capture_predictions", False),
            )
            results = {
                "test_results": test_results,
                "test_details": getattr(model, "last_test_details", None),
                "parameter_counts": parameter_counts,
            }
            if getattr(args, "capture_predictions", False):
                results["test_artifacts"] = getattr(model, "last_test_artifacts", None)
        else:
            raise ValueError
    except Exception:
        trainer.logger.info(traceback.format_exc())
        if getattr(args, "raise_exceptions", False):
            raise
    return results

if __name__=='__main__':
    if '--submission-run' in sys.argv:
        from scripts.run_hybstex import main as run_final_hybstex

        forwarded = [arg for arg in sys.argv[1:] if arg != '--submission-run']
        raise SystemExit(run_final_hybstex(forwarded))

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
    parser.add_argument('--variant', "-v", default=None, type=str, help='which variant of model to use. pred/cls/bias')
    parser.add_argument('--phase3_mode', default="original", choices=["original", "joint_separated"],
                    type=str, help='phase-3 training mode')
    parser.add_argument(
        '--ablation_mode',
        default="original",
        choices=[
            "original",
            "ungated_bias",
            "floor_gated",
            "dual_event_residual",
            "event_weighted_base",
            "event_weighted_ungated_end_to_end",
            "event_weighted_ungated_residual",
            "event_weighted_original",
            "event_weighted_dual_residual",
            "event_weighted_dual_ungated",
            "boosted_gate",
            "event_weighted_boosted_gate",
            "expanded_base_additive",
            "node_bias_only",
            "flow_bias_only",
        ],
        type=str,
        help='ablation mode to use',
    )
    parser.add_argument(
        '--bias_param_scope',
        default="legacy",
        choices=["legacy", "head_only"],
        type=str,
        help='legacy groups every parameter with "bias" in its name; head_only groups only residual heads',
    )
    parser.add_argument('--gate_floor', default=0.0, type=float,
                    help='minimum residual gate for floor_gated mode')
    parser.add_argument('--boost_gate_scale', default=1.0, type=float,
                    help='nonnegative multiplier for boosted_gate residual amplification')
    parser.add_argument('--classification_loss_weight', default=1.0, type=float,
                    help='weight for BCE when it participates in the current phase')
    parser.add_argument('--event_loss_weight', default=0.0, type=float,
                    help='additional MAE weight on evs_90 points for event-weighted modes')
    parser.add_argument('--event_percentile', default=None, type=float,
                    help='derive event labels in memory from this training-target percentile')
    parser.add_argument('--event_mask_protocol', default='legacy_raw_p90_v1', type=str,
                    choices=['legacy_raw_p90_v1', 'train_all_node_flow_p90_valid_v2'],
                    help='event-mask protocol; corrected v2 is an explicit opt-in')
    parser.add_argument('--event_label_source', default='legacy_file_unverified', type=str,
                    choices=['legacy_file_unverified', 'file_verified', 'generated'],
                    help='whether v2 verifies saved raw labels before using corrected masks')
    parser.add_argument('--start_phase', default="pred", choices=["pred", "cls", "bias", "pred_2"],
                    type=str, help='first phase to train')
    parser.add_argument('--stop_after_phase', default="pred_2", choices=["pred", "cls", "bias", "pred_2"],
                    type=str, help='last phase to train')
    parser.add_argument('--evaluation_phase', default=None, choices=["pred", "cls", "bias", "pred_2"],
                    type=str, help='explicit phase to use for test evaluation')
    parser.add_argument('--capture_predictions', action='store_true', default=None,
                    help='store CPU float32 prediction, target, event, and valid test artifacts')
    parser.add_argument('--raise_exceptions', action='store_true', default=None,
                    help='re-raise supervisor exceptions after logging them')
    parser.add_argument('--data_dir', default=None, type=str, help='override dataset root directory')
    parser.add_argument('--graph_file', default=None, type=str, help='override graph file path')
    parser.add_argument('--epochs', default=None, type=int, help='override number of training epochs per phase')
    parser.add_argument('--num_epochs', default=None, type=int, help='override scheduler epoch count')
    parser.add_argument('--batch_size', default=None, type=int, help='override training batch size')
    parser.add_argument('--test_batch_size', default=None, type=int, help='override validation/test batch size')
    parser.add_argument('--device', default=None, type=str, help='override torch device')
    parser.add_argument('--max_train_batches', default=None, type=int, help='optional train batch limit for smoke tests')
    parser.add_argument('--max_eval_batches', default=None, type=int, help='optional validation/test batch limit for smoke tests')

    # parser.add_argument('--input_length', default=0, type=int, help='# of samples to use for context')
    args = parser.parse_args()
    print(f'Starting experiment with configurations in {args.config_filename}...')
    
    with open(args.config_filename, encoding='utf-8') as config_handle:
        configs = yaml.safe_load(config_handle)
    
    configs['S_Loss'] = args.S_Loss
    configs['T_Loss'] = args.T_Loss
    configs['mode'] = args.mode
    configs['best_path'] = args.best_path
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
    configs['phase3_mode'] = args.phase3_mode
    configs['ablation_mode'] = args.ablation_mode
    configs['bias_param_scope'] = args.bias_param_scope
    configs['gate_floor'] = args.gate_floor
    configs['boost_gate_scale'] = args.boost_gate_scale
    configs['classification_loss_weight'] = args.classification_loss_weight
    configs['event_loss_weight'] = args.event_loss_weight
    configs['event_percentile'] = args.event_percentile
    configs['event_mask_protocol'] = args.event_mask_protocol
    configs['event_label_source'] = args.event_label_source
    configs['start_phase'] = args.start_phase
    configs['stop_after_phase'] = args.stop_after_phase
    if args.evaluation_phase is not None:
        configs['evaluation_phase'] = args.evaluation_phase
    if args.capture_predictions is not None:
        configs['capture_predictions'] = args.capture_predictions
    if args.raise_exceptions is not None:
        configs['raise_exceptions'] = args.raise_exceptions
    configs['max_train_batches'] = args.max_train_batches
    configs['max_eval_batches'] = args.max_eval_batches
    if args.data_dir is not None:
        configs['data_dir'] = args.data_dir
    if args.graph_file is not None:
        configs['graph_file'] = args.graph_file
    if args.epochs is not None:
        configs['epochs'] = args.epochs
        if args.num_epochs is None:
            configs['num_epochs'] = args.epochs
    if args.num_epochs is not None:
        configs['num_epochs'] = args.num_epochs
    if args.batch_size is not None:
        configs['batch_size'] = args.batch_size
    if args.test_batch_size is not None:
        configs['test_batch_size'] = args.test_batch_size
    if args.device is not None:
        configs['device'] = args.device
    
    # configs['input_length'] = args.input_length
    # experimentName = "pred_" + str(args.input_length) + "_"
    experimentName = "pred_"
    if args.S_Loss == 1:
        experimentName += "+S"

    if args.T_Loss == 1:
        experimentName += "+T"
    if args.phase3_mode != "original":
        experimentName += f"_{args.phase3_mode}"
    if args.ablation_mode != "original":
        experimentName += f"_{args.ablation_mode}"
    experimentName += f"_seed={args.seed}"
    
    configs["experimentName"] = experimentName
    print(f'Starting experiment with configurations {configs}...')
    args = argparse.Namespace(**configs)
    model_supervisor(args)
