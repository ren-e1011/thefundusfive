
from pathlib import Path 
import os, sys
sys.path.append(str(Path.cwd().parent)) # for pythonpath 
sys.path.append(str(Path.cwd()))
import json
from datetime import timedelta
from typing import Iterable, Optional

import torch
from torch.utils.tensorboard import SummaryWriter
from utils import configure_optimizer, accuracy 

import RETFound_MAE.models_vit as models_vit 
from RETFound_MAE.util.pos_embed import interpolate_pos_embed
from timm.models.layers import trunc_normal_



import RETFound_MAE.util.lr_decay as lrd
import RETFound_MAE.util.misc as misc
from RETFound_MAE.util.misc import NativeScalerWithGradNormCount as NativeScaler


from utils import adjust_learning_rate

from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
import time
from math import isfinite

from test import evaluate

import wandb 

# with slight mod from RETFound_MAE engine_finetune train_one_epoch 
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer, 
                    device: torch.device, epoch: int, loss_scaler, logger, max_norm: float = 0, 
                    training_params = None, scheduler: torch.optim.lr_scheduler = None,
                    log_writer=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = training_params.logging.print_freq 

    accum_iter = training_params.accum_iter if training_params is not None else 1

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        # if data_iter_step % accum_iter == 0:
        #     adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, training_params)


        # with torch.autocast(device_type=device, dtype=torch.bfloat16): RuntimeError: User specified an unsupported autocast device_type 'cpu' - maybe 3.8 is not suited for cpu
        outputs = model(samples)
        loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # loss /= accum_iter
        # loss_scaler(loss, optimizer, clip_grad=max_norm,
        #             parameters=model.parameters(), create_graph=False,
        #             update_grad=(data_iter_step + 1) % accum_iter == 0)
        
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value, epoch_1000x) #MOD
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

        acc = accuracy(outputs, targets)
        logger.log({"train_acc":acc, "train_loss":loss,'learning_rate': max_lr})

    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# with slight mod from RETFound_MAE util misc save_model
def save_model(model, params = None, optimizer=None,loss_scaler=None,epoch=-1,k=-1):
    output_dir = Path(f"{params.out_dir}/Fold_{k}") if params else '.'
    task = params.task if params else ''
    epoch_str = str(epoch)
    fold = str(k)

    if loss_scaler is not None:
        checkpoint_paths = [task+'_fold_'+fold+'_ckpt-best.pth']
        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict()
            }
            torch.save(to_save, checkpoint_path) # equivalent to save state dict
    else:
        print("Checkpointing wo loss")
        client_state = {'epoch':epoch}
        model.save_checkpoint(save_dir=output_dir, tag="checkpoint-%s" % epoch_str, client_state=client_state)

def train_retfund_fives(dataset, training_params, data_params, device='cpu'):

    batch_size = training_params.batch_size
    n_classes = training_params.n_classes

    k = 5
    kFold = KFold(n_splits=k,shuffle=True) 

    for fold, (train_ids, val_ids) in enumerate(kFold.split(dataset)):

        print('====================')
        print('Fold ',fold)
        print('====================')

        # Sample elements randomly from a given list of ids, no replacement.
        train_sampler = SubsetRandomSampler(train_ids)
        val_sampler = SubsetRandomSampler(val_ids)

        dataloader_train = DataLoader(
            dataset, sampler=train_sampler,
            batch_size=batch_size,
        )

        dataloader_val = DataLoader(
            dataset, sampler=val_sampler,
            batch_size=batch_size,
            )
    # =============================================================================
    # Model load 
    # =============================================================================

    # "Load the model and weights" snippet @ https://github.com/rmaphoh/RETFound_MAE/tree/main
        model = models_vit.__dict__['vit_large_patch16'](
            img_size = data_params.out_size, # config.data.input_size 
            num_classes=n_classes, # mod
            drop_path_rate=0.2,
        )

        # load RETFound weights
        checkpoint = torch.load(training_params.checkpoint, map_location='cpu')
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model) # ? 

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)

        # if args.global_pool 
        assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # manually initialize fc layer
        trunc_normal_(model.head.weight, std=2e-5)

    # snippet from RETFound_MAE.main_finetune.main

        model.to(device)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # print("Model = %s" % str(model))
        print('number of params (M): %.2f' % (n_parameters / 1.e6))
    # =============================================================================
    # Optimizer, Scheduler
    # =============================================================================

        # TODO mod with scheduling
        # build optimizer with layer-wise lr decay (lrd)
        param_groups = lrd.param_groups_lrd(model, training_params.weight_decay,
            no_weight_decay_list=model.no_weight_decay(),
            layer_decay=training_params.layer_decay
        )
        # _lr = 1e-3 * 16/256
        # optimizer = torch.optim.AdamW(param_groups,lr =  _lr)
        optimizer, scheduler = configure_optimizer(model=model,data_config=data_params, train_config=training_params)

        # optimizer = torch.optim.AdamW(model.parameters()) # MOD
        loss_scaler = NativeScaler(device) # black box scaling the lr 

    # =============================================================================
    # Learning criterion
    # =============================================================================

        criterion = torch.nn.CrossEntropyLoss()

    # =============================================================================
    # Checkpointing 
    # =============================================================================    
        output_dir = training_params.checkpointing.out_dir
        log_dir = training_params.logging.log_dir + f"Fold_{fold}/"
        task = training_params.logging.task
        os.makedirs(log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=log_dir+task)

        print(f"Start training for {training_params.max_epch} epochs")
        start_time = time.time()
        max_accuracy = 0.0
        max_auc = 0.0

    # =============================================================================
    # Logging
    # =============================================================================
    
        wblogger = wandb.init(
            project="fundus5",
            name=training_params.logging.name,
            notes=f"batch_size {training_params.batch_size}",
            config={"metric":{"goal":"maximize","name":"val_acc"}})

    # =============================================================================
    # Training loop 
    # =============================================================================
        for epoch in range(training_params.max_epch):
            
            train_stats = train_one_epoch(
                model = model, criterion = criterion, data_loader=dataloader_train,
                optimizer=optimizer, scheduler = scheduler, device=device, epoch=epoch, loss_scaler=loss_scaler,
                log_writer=log_writer, training_params = training_params, logger = wblogger
            )

            # wblogger.log({"epoch_train_acc":train_stats[''], "epoch_train_loss":,})

            val_stats,val_auc_roc = evaluate(dataloader_val, model, device,out_dir = log_dir,epoch = epoch, mode='val',n_classes=n_classes, logger=wblogger)
            
        
    # =============================================================================
    # Save model 
    # =============================================================================
            if max_auc<val_auc_roc:
                max_auc = val_auc_roc
                print(f"Saving model to {output_dir}")
                if output_dir:
                    # misc save model
                    
                    save_model(
                        model=model, params=training_params.checkpointing, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch=epoch,k=fold)

            if log_writer is not None:
                log_writer.add_scalar('perf/val_acc1', val_stats['acc1'], epoch)
                log_writer.add_scalar('perf/val_auc', val_auc_roc, epoch)
                log_writer.add_scalar('perf/val_loss', val_stats['loss'], epoch)
                
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            'epoch': epoch,
                            'n_parameters': n_parameters}

            if output_dir and log_writer is not None:
                log_writer.flush()
                with open(os.path.join(output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")
        
        wblogger.finish()
                    
        total_time = time.time() - start_time
        total_time_str = str(timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
        
