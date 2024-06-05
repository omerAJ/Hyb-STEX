


import sys
sys.path.append('.')
sys.path.append('..')
import torch
from lib.dataloader import get_dataloader
import argparse
import yaml
import time
from model.vision_transformer import VisionTransformer, VisionTransformerPredictor
import torch.nn.functional as F 
from model.vision_transformer_utils import init_opt, apply_masks_targets, repeat_interleave_batch

import numpy as np
_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', '-cf', default='configs/BJTaxi.yaml', 
                        type=str, help='the configuration to use')

    args = parser.parse_args()
    print(f'Starting experiment with configurations in {args.config_filename}...')

    time.sleep(3)
    configs = yaml.load(
        open(args.config_filename), 
        Loader=yaml.FullLoader
    )
    args = argparse.Namespace(**configs)
    start_epoch = args.start_epoch
    num_epochs = args.num_epochs

    dataloader = get_dataloader(
        data_dir=args.data_dir, 
        dataset=args.dataset, 
        batch_size=args.batch_size, 
        test_batch_size=args.test_batch_size,
        scalar_type='Standard'
    )

    train_loader = dataloader['train']
    val_loader = dataloader['val']
    test_loader = dataloader['test']



    ## init models

    encoder = VisionTransformer(
        img_size=(args.row, args.col),
        patch_size=1,
        in_chans=70,
        embed_dim=8,
        predictor_embed_dim=None,
        depth=1,
        predictor_depth=None,
        num_heads=1,
        mlp_ratio=2,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.4,
        attn_drop_rate=0.4,
        drop_path_rate=0.3,
        norm_layer=torch.nn.LayerNorm,
        init_std=0.02
    )
    predictor = VisionTransformerPredictor(
        img_size=(args.row, args.col),
        embed_dim=8,
        predictor_embed_dim=8//2,
        depth=1,
        num_heads=1,
        mlp_ratio=2,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.4,
        attn_drop_rate=0.4,
        drop_path_rate=0.3,
        norm_layer=torch.nn.LayerNorm,
        init_std=0.02
    )

    import copy
    target_encoder = copy.deepcopy(encoder)

    wd = 0.04
    final_wd = 0.4
    start_lr = 0.00002
    final_lr = 1.0e-06
    lr = 0.0001
    ipe = len(train_loader)
    warmup = 10
    ipe_scale = 1.0
    use_bfloat16 = True
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
            encoder=encoder,
            predictor=predictor,
            wd=wd,
            final_wd=final_wd,
            start_lr=start_lr,
            ref_lr=lr,
            final_lr=final_lr,
            iterations_per_epoch=ipe,
            warmup=warmup,
            num_epochs=num_epochs,
            ipe_scale=ipe_scale,
            use_bfloat16=use_bfloat16)

    """dont backprop to target_encoder parameters, update it with ema"""
    for p in target_encoder.parameters():
        p.requires_grad = False
    ema = [0.996, 1.0]
    ipe = len(train_loader)
    ipe_scale = 1.0
    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale)
                            for i in range(int(ipe*num_epochs*ipe_scale)+1))
    
    # for _ in range(240): print("momentum_scheduler: ", next(momentum_scheduler))


    import logging
    log_freq = 100
    checkpoint_freq = 2
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logger = logging.getLogger()
    from model.ijepa_utils import AverageMeter, CSVLogger, gpu_timer, grad_logger


    # -- make csv_logger
    rank=0
    world_size=1
    import os
    tag = r"jepa"
    folder = r"E:\estudy\ST-SSL\code\ST-SSL\logs\test"
    log_file = os.path.join(folder, f'{tag}_r{rank}.csv')
    save_path = os.path.join(folder, f'{tag}' + '-ep{epoch}.pth.tar')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    log_file = os.path.join(folder, f'{tag}-log.csv')
    csv_logger = CSVLogger(log_file,
                            ('%d', 'epoch'),
                            ('%d', 'itr'),
                            ('%.5f', 'loss'),
                            ('%d', 'time (ms)'))
   

    def save_checkpoint(epoch):
        save_dict = {
            'encoder': encoder.state_dict(),
            'predictor': predictor.state_dict(),
            'target_encoder': target_encoder.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'loss': loss_meter.avg,
            'batch_size': args.batch_size,
            'world_size': world_size,
            'lr': lr
        }
        if rank == 0:
            torch.save(save_dict, latest_path)
            if (epoch + 1) % checkpoint_freq == 0:
                torch.save(save_dict, save_path.format(epoch=f'{epoch + 1}'))
                
    for epoch in range(start_epoch, num_epochs):
        
        logger.info('Epoch %d' % (epoch + 1))
        loss_meter = AverageMeter()
        test_loss_meter = AverageMeter()
        time_meter = AverageMeter()
        # print("\n\n\nbefore enmerating loader")
        for train_itr, (data, target) in enumerate(train_loader):
            b = torch.randint(0, 32, (1,))
            # print("data.shape: ", data.shape)   ## torch.Size([32, 35, 200, 2])
            # data = data[:, b, :, :].squeeze(1)  ## select just one graph
            B, T, N, D = data.size()
            data = data.transpose(1, 2).reshape(B, N, -1)
            # print("data.shape: ", data.shape)   ## torch.Size([32, 200, 2])
            B, N, D = data.size()
            data = data.view(B, args.row, args.col, D).to(args.device) ## reshape to 2D grid 
            # print("data.shape: ", data.shape)   ## torch.Size([32, 20, 10, 2])
            
            
            def generateMasks(data):
                # :param data: tensor of shape [B, R, C, D]
                # :returns: (data, 1x masks_enc, 4x masks_pred)
                import numpy as np
                # grid_pd = np.load(r"E:\estudy\ST-SSL\code\ST-SSL\data\NYCTaxi\grid_pd.npy")
                # _pd = grid_pd.flatten()
                B, R, C, D = data.size()

                # Initialize masks
                masks_enc = torch.zeros(B, R, C, dtype=torch.uint8)
                masks_pred = torch.zeros(4, B, R, C, dtype=torch.uint8)

                ## select size of context and target masks before sample loop.
                ## grid_size: 20x10
                
                masks_enc = masks_enc.flatten(1)
                masks_pred = masks_pred.flatten(2)
                # ctxt_size = torch.randint(50, 100, (1,)).item()       ## low (inclusive), high (exclusive)
                # trgt_size = torch.randint(50//4, 100//4, (1,)).item()  
                ctxt_size = torch.randint(300, 800, (1,)).item()       ## low (inclusive), high (exclusive)
                leftOutNodes = R*C - ctxt_size
                trgt_size = torch.randint(leftOutNodes//2, leftOutNodes, (1,)).item()  
                # print(f"ctxt_size: {ctxt_size}, trgt_size: {trgt_size}")
                # print(f"masks_enc.shape: {masks_enc.shape}, masks_pred.shape: {masks_pred.shape}")  ##m asks_enc.shape: torch.Size([32, 200]), masks_pred.shape: torch.Size([4, 32, 200])
                _try=0
                for b in range(B):
                    # pd = _pd.copy()
                    # print(f"\n\ncxtz: {ctxt_size}")
                    # ctxt_indices = np.random.choice(R*C, size=ctxt_size, replace=False, p=pd)
                    ctxt_indices = np.random.choice(R*C, size=ctxt_size, replace=False)
                    available_indices = np.setdiff1d(np.arange(R*C), ctxt_indices)
                    # print(f"mask_enc.shape: {masks_enc.shape} ")
                    # print(f"mask_enc.shape: {masks_enc.shape} ")
                    masks_enc[b, ctxt_indices] = 1 
                    # pd[ctxt_indices] = 0           ## set ctxt indices to 0 so that those nodes are not repeated in trgt masks
                    # pd /= pd.sum()
                    # Generate four prediction masks (can be overlapping with each other, but not with context mask)
                    for i in range(4):
                        # Smaller random sizes for prediction masks
                        trgt_indices = np.random.choice(available_indices, size=trgt_size, replace=False)
                        # trgt_indices = np.random.choice(R*C, size = trgt_size, replace=False, p=pd) 
                        masks_pred[i, b, trgt_indices] = 1
                masks_enc = masks_enc.view(B, 1, R*C)
                masks_pred = masks_pred.view(4, B, R*C)
                return (data, masks_enc, masks_pred.transpose(0, 1))  
            
            """
            def generateMasks(data):
                
                # :param data: tensor of shape [B, R, C, D]
                # :returns: (data, 1x masks_enc, 4x masks_pred)
                
                B, R, C, D = data.size()

                # Initialize masks
                masks_enc = torch.zeros(B, R, C, dtype=torch.uint8)
                masks_pred = torch.zeros(4, B, R, C, dtype=torch.uint8)

                ## select deltas before sample loop. 
                ## grid_size: 20x10
                delta_h_ctxt = torch.randint(9, 13, (1,)).item()       ## low (inclusive), high (exclusive)
                delta_w_ctxt = torch.randint(6, 8, (1,)).item()  
                delta_h_trgt = torch.randint(4, 7, (1,)).item()  
                delta_w_trgt = torch.randint(3, 5, (1,)).item()  
                
                for b in range(B):
                    
                    # print(f"R: {R}, delta_h_ctxt: {delta_h_ctxt}" )
                    h1 = torch.randint(0, R-delta_h_ctxt, (1,))
                    if h1 + delta_h_ctxt > R:
                        h1 = R - delta_h_ctxt
                    w1 = torch.randint(0, C-delta_w_ctxt, (1,))
                    if w1 + delta_w_ctxt > C:
                        w1 = C - delta_w_ctxt
                    
                    # Set the encoding mask
                    masks_enc[b, h1:h1+delta_h_ctxt, w1:w1+delta_w_ctxt] = 1

                    # Generate four prediction masks (can be overlapping with each other, but not with context mask)
                    for i in range(4):
                        n=0
                        while True:
                            if n>2000:
                                # print(f"cant find valid mask n={n}, stuck...")
                                return None    
                            # Smaller random sizes for prediction masks
                            h1 = torch.randint(0, R-delta_h_trgt, (1,))
                            if h1 + delta_h_trgt > R:
                                h1 = R - delta_h_trgt
                            w1 = torch.randint(0, C-delta_w_trgt, (1,))
                            if w1 + delta_w_trgt > C:
                                w1 = C - delta_w_trgt

                            # Create a temporary mask to check overlap
                            temp_mask = torch.zeros(R, C, dtype=torch.bool)
                            temp_mask[h1:h1+delta_h_trgt, w1:w1+delta_w_trgt] = 1

                            # Check if it overlaps with the encoding mask
                            if torch.any(masks_enc[b] & temp_mask):
                                n+=1
                                continue  # Overlaps, try again

                            # No overlap, set this mask
                            masks_pred[i, b] = temp_mask
                            break

                return (data, masks_enc, masks_pred.transpose(0, 1))  
            """
            data = generateMasks(data)
            if data is None:
                # print(f"generateMasks returned None for Epoch, itr: {epoch, train_itr}")
                continue
            else:
                imgs, masks_enc, masks_pred = data
            # print("generated masks: \n", "imgs.shape: ", imgs.shape, " masks_enc.shape: ", masks_enc.shape, " masks_pred.shape: ", masks_pred.shape, "\n\n\n")
            imgs = imgs.permute(0, 3, 1, 2)  ## [B, R, C, D] -> [B, D, R, C]
            masks_pred = masks_pred.flatten(2) ## [B, 4, R, C] -> [B, 4, R*C]
            masks_enc = masks_enc.flatten(1).unsqueeze(1) ## [B, R, C] -> [B, 1, R*C]
            # print("generated masks: \n", "imgs.shape: ", imgs.shape, " masks_enc.shape: ", masks_enc.shape, " masks_pred.shape: ", masks_pred.shape, "\n\n\n")
            ## imgs.shape:  torch.Size([32, 2, 20, 10])  masks_enc.shape:  torch.Size([32, 200])  masks_pred.shape:  torch.Size([32, 4, 200])
            
            """flattened imgs and masks using .flatten() method."""
            """visualizing the context and target masks on the image"""
            """
            masks_enc = masks_enc.view(B, 1, args.row, args.col).squeeze(1)
            masks_pred = masks_pred.view(B, 4, args.row, args.col)
            def count_unique_elements(masks_enc):
                import numpy as np
                unique_elements, counts = np.unique(masks_enc, return_counts=True)
                return dict(zip(unique_elements, counts))
            for b in range(32):
                for i in range(4): masks_enc[b, :, :][masks_pred[b, i, :, :] == 1] = 2
                print("masked tensor: \n\n", masks_enc[b, :, :])    
                print(count_unique_elements(masks_enc[b, :, :].cpu().numpy()))
                # mask_ctxt = masks_enc.clone()
                # for i in range(4): 
                #     masks_enc = mask_ctxt.clone()
                #     masks_enc[b, :, :][masks_pred[b, i, :, :] == 1] = 2
                #     print(f"\n\nmask tensor (mask_{i}): \n\n", masks_enc[b, :, :])    
                #     print(count_unique_elements(masks_enc[b, :, :].cpu().numpy()))
            """
            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()
                def forward_target():    ## mask target tokens after encoding
                    with torch.no_grad():  #sg
                        h = target_encoder(imgs, masks=None)  ## VisionTransformer  masks_enc=None
                        h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
                        B = len(h)
                        # print("h.shape: ", h.shape)    ## torch.Size([32, 200, 256])
                        # -- all tokens updated, now create 4 targets
                        # -- create targets (masked regions of h)
                        # print("\n\nh.shape (before apply_masks_targets): ", h.shape, "mask_pred.shape: ", masks_pred.shape, "\n\n")
                        h = apply_masks_targets(h, masks_pred)
                        # print("\n\nh.shape (after apply_masks_targets): ", h.shape, "\n\n") 
                        # h = repeat_interleave_batch(h, B, repeat=len(masks_enc))    ## does nothing as len(x)==B and repeat==1
                        return h

                def forward_context():    ## mask context tokens before encoding
                    z = encoder(imgs, masks=masks_enc)  ## VisionTransformer
                    # print("(before encoder) imgs.shape: ", imgs.shape, " masks_enc.shape: ", masks_enc.shape, "(after encoder) z.shape: ", z.shape)
                    """input to predictor is z: [32, 45, 256]"""
                    z = predictor(z, masks_enc, masks_pred)   ## VisionTransformerPredictor
                    return z

                def loss_fn(z, h):
                    loss = F.smooth_l1_loss(z, h)
                    # loss = AllReduce.apply(loss)
                    return loss

                # Step 1. Forward
                # with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                h = forward_target()
                z = forward_context()
                loss = loss_fn(z, h)

                #  Step 2. Backward & step
                loss.backward()
                optimizer.step()

                grad_stats = grad_logger(encoder.named_parameters())
                optimizer.zero_grad()

                # Step 3. momentum update of target encoder
                
                with torch.no_grad():
                    m = next(momentum_scheduler)
                    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                        param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

                return (float(loss), _new_lr, _new_wd, grad_stats)
            
            (loss, _new_lr, _new_wd, grad_stats), etime = gpu_timer(train_step)
            loss_meter.update(loss)
            time_meter.update(etime)

            ## eval time
            def test_step():
                for itr, (data, target) in enumerate(test_loader):
                    # data = data[:, 0, :, :].squeeze(1)  ## select just one graph
                    # print("data.shape: ", data.shape)   ## torch.Size([32, 200, 2])
                    B, T, N, D = data.size()
                    data = data.transpose(1, 2).reshape(B, N, -1)
                    B, N, D = data.size()
                    data = data.view(B, args.row, args.col, D).to(args.device) ## reshape to 2D grid 
                    data = generateMasks(data)
                    if data is None:
                        print(f"generateMasks returned None for Epoch, itr: {epoch, train_itr}")
                        continue
                    else:
                        imgs, masks_enc, masks_pred = data
                    imgs = imgs.permute(0, 3, 1, 2)  ## [B, R, C, D] -> [B, D, R, C]
                    masks_pred = masks_pred.flatten(2) ## [B, 4, R, C] -> [B, 4, R*C]
                    masks_enc = masks_enc.flatten(1).unsqueeze(1) ## [B, R, C] -> [B, 1, R*C]
                    with torch.no_grad():
                        def forward_target():    ## mask target tokens after encoding
                            with torch.no_grad():  #sg
                                h = target_encoder(imgs, masks=None)  ## VisionTransformer  masks_enc=None
                                h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
                                B = len(h)
                                # print("h.shape: ", h.shape)    ## torch.Size([32, 200, 256])
                                # -- all tokens updated, now create 4 targets
                                # -- create targets (masked regions of h)
                                # print("\n\nh.shape (before apply_masks_targets): ", h.shape, "mask_pred.shape: ", masks_pred.shape, "\n\n")
                                h = apply_masks_targets(h, masks_pred)
                                # print("\n\nh.shape (after apply_masks_targets): ", h.shape, "\n\n") 
                                # h = repeat_interleave_batch(h, B, repeat=len(masks_enc))    ## does nothing as len(x)==B and repeat==1
                                return h

                        def forward_context():    ## mask context tokens before encoding
                            z = encoder(imgs, masks=masks_enc)  ## VisionTransformer
                            # print("(before encoder) imgs.shape: ", imgs.shape, " masks_enc.shape: ", masks_enc.shape, "(after encoder) z.shape: ", z.shape)
                            """input to predictor is z: [32, 45, 256]"""
                            z = predictor(z, masks_enc, masks_pred)   ## VisionTransformerPredictor
                            return z

                        def loss_fn(z, h):
                            loss = F.smooth_l1_loss(z, h)
                            # loss = AllReduce.apply(loss)
                            return loss

                        # Step 1. Forward
                        # with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        h = forward_target()
                        z = forward_context()
                        test_loss = loss_fn(z, h)
                        test_loss_meter.update(test_loss)

                    return test_loss_meter
            
            # -- Logging
            import numpy as np
            def log_stats():
                csv_logger.log(epoch + 1, train_itr, loss, etime)
                if (train_itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    test_loss_meter = test_step()
                    
                    logger.info('[%d, %5d] loss: %.5f '
                                'test_loss: %.5f '
                                '[wd: %.2e] [lr: %.2e] '
                                '[mem: %.2e] '
                                '(%.1f ms)'
                                % (epoch + 1, train_itr,
                                    loss_meter.avg,
                                    test_loss_meter.avg,
                                    _new_wd,
                                    _new_lr,
                                    torch.cuda.max_memory_allocated() / 1024.**2,
                                    time_meter.avg))

                    if grad_stats is not None:
                        logger.info('[%d, %5d] grad_stats: [%.2e %.2e] (%.2e, %.2e)'
                                    % (epoch + 1, train_itr,
                                        grad_stats.first_layer,
                                        grad_stats.last_layer,
                                        grad_stats.min,
                                        grad_stats.max))

            log_stats()

            assert not np.isnan(loss), 'loss is nan'

        # -- Save Checkpoint after every epoch
        logger.info('avg. loss %.3f' % loss_meter.avg)
        save_checkpoint(epoch+1)


if __name__ == "__main__":
    main()
