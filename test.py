from pathlib import Path 
import os, sys
sys.path.append(str(Path.cwd().parent)) # for pythonpath 
sys.path.append(str(Path.cwd()))
import csv
import numpy as np

import torch
import RETFound_MAE.models_vit as models_vit 
from RETFound_MAE.util.pos_embed import interpolate_pos_embed
from timm.models.layers import trunc_normal_

from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, average_precision_score,multilabel_confusion_matrix
from timm.utils import accuracy
import matplotlib.pyplot as plt

import torch.nn.functional as F
from torch.nn import Softmax
from RETFound_MAE.util.misc import MetricLogger

from utils import misc_measures
import pickle
import wandb

# slight mods from RETFound_MAE engine_finetune evaluate
@torch.no_grad()
def evaluate(data_loader, model, device, out_dir, logger = None, k='', mode='val', n_classes=4):
    task = out_dir 
    
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'
    
    if not os.path.exists(task):
        os.makedirs(task)

    prediction_decode_list = []
    prediction_list = []
    true_label_decode_list = []
    true_label_onehot_list = []
    
    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]

        true_label=F.one_hot(target.to(torch.int64), num_classes=n_classes)

        # compute output
        # with torch.autocast(device_type=device):
        output = model(images)
        loss = criterion(output, target)
        prediction_softmax = Softmax(dim=1)(output)
        _,prediction_decode = torch.max(prediction_softmax, 1)
        _,true_label_decode = torch.max(true_label, 1)

        prediction_decode_list.extend(prediction_decode.cpu().detach().numpy())
        true_label_decode_list.extend(true_label_decode.cpu().detach().numpy())
        true_label_onehot_list.extend(true_label.cpu().detach().numpy())
        prediction_list.extend(prediction_softmax.cpu().detach().numpy())

        acc1,acc2 = accuracy(output, target, topk=(1,2)) # leave topk as 1 mod from topk=(1,2)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
    # gather the stats from all processes
    true_label_decode_list = np.array(true_label_decode_list)
    prediction_decode_list = np.array(prediction_decode_list)
    confusion_matrix = multilabel_confusion_matrix(true_label_decode_list, prediction_decode_list,labels=[i for i in range(n_classes)])
    acc, sensitivity, specificity, precision, G, F1, mcc = misc_measures(confusion_matrix)
    
    auc_roc = roc_auc_score(true_label_onehot_list, prediction_list,multi_class='ovr',average='macro')
    auc_pr = average_precision_score(true_label_onehot_list, prediction_list,average='macro')          
            
    
    print('Sklearn Metrics - Acc: {:.4f} AUC-roc: {:.4f} AUC-pr: {:.4f} F1-score: {:.4f} MCC: {:.4f}'.format(acc, auc_roc, auc_pr, F1, mcc)) 
    
    
    results_path = task+'_metrics_{}{}.csv'.format(mode,k)
    
    with open(results_path,mode='a',newline='',encoding='utf8') as cfa:
        wf = csv.writer(cfa)
        data2=[[acc,sensitivity,specificity,precision,auc_roc,auc_pr,F1,mcc,metric_logger.loss]]
        for i in data2:
            wf.writerow(i)
            
    # TypeError: 'numpy.ndarray' object is not callable
    # if mode=='test':
    #     cm = confusion_matrix(actual_vector=true_label_decode_list, predict_vector=prediction_decode_list)
    #     cm.plot(cmap=plt.cm.Blues,number_label=True,normalized=True,plot_lib="matplotlib")
    #     plt.savefig(task+'confusion_matrix_test.jpg',dpi=600,bbox_inches ='tight')
    
    if logger: 
        logger.log({f"{mode}_loss":loss, f"{mode}_accuracy":acc, 'F1-score':F1, 'AUC_ROC':auc_roc, 'AUC_Precision':auc_pr})
    else:
        print(f"{mode}_loss:",loss, f"{mode}_accuracy:",acc, 'F1-score:',F1, 'AUC_ROC:',auc_roc, 'AUC_Precision:',auc_pr)

    if mode == 'test':
        return loss, acc, F1, auc_roc, auc_pr 

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()},auc_roc


def test_retfund_fives(dataset, training_params, data_params, device='cpu', folds=2, ckpt_path = "./",ckpt_filestart="retfundfives_finetune__fold_",log_dir = './test_logs',task=""):
    test_ids = [i for i in range(len(dataset))]
    test_sampler = SubsetRandomSampler(test_ids)
    n_classes = training_params.n_classes

    data_loader_test = DataLoader(
        dataset, sampler=test_sampler,
        batch_size= training_params.batch_size
    )

    metric_dict = {
    'loss_mean' : [],
    'acc_mean' : [],
    'f1_mean' : [],
    'aucroc_mean' : [],
    'aucpr_mean' : []
    }

    # wblogger = wandb.init(
    #         project="fundus5",
    #         name=training_params.logging.name,
    #         notes=f"batch_size {training_params.batch_size}",
    #         config={"metric":{"goal":"maximize","name":"val_acc"}})

    # load model similar to train TODO 

    # state_dict_best = torch.load(task+'checkpoint-best.pth', map_location='cpu')
    # model.load_state_dict(state_dict_best['model'])
    # test_stats,auc_roc = evaluate(data_loader_test, model, device, task,epoch=0, mode='test',num_class=n_classes)

    for k in range(folds):
        mdl_at = ckpt_path + ckpt_filestart + str(k) + '_ckpt-best.pth'

        model = models_vit.__dict__['vit_large_patch16'](
            img_size = data_params.out_size, # config.data.input_size 
            num_classes=n_classes, # mod
            drop_path_rate=0.2,
        )

        # load RETFound weights
        checkpoint = torch.load(mdl_at, map_location='cpu')
        checkpoint_model = checkpoint['model']
  
        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)

        # if args.global_pool 
        assert set(msg.missing_keys) == set() #empty set 

        # manually initialize fc layer
        # trunc_normal_(model.head.weight, std=2e-5)
        model.eval()


        
        
        loss_fold, acc_fold, f1_fold, aucroc_fold, aucpr_fold = evaluate(data_loader_test,model,device,out_dir=log_dir,k=str(k), mode='test',n_classes=n_classes) 

        metric_dict['loss_mean'].append(loss_fold)
        metric_dict['acc_mean'].append(acc_fold)
        metric_dict['f1_mean'].append(f1_fold)
        metric_dict['aucroc_mean'].append(aucroc_fold)
        metric_dict['aucpr_mean'].append(aucpr_fold)

    # wblogger.finish()
    # mod to pickle 
    filename = f"{log_dir}+/+{training_params.checkpointing.task}+{data_params.type}+_+{training_params.type}_{folds}_folds.pkl"
    with open(filename,"wb") as outfile:
        pickle.dump(metric_dict, outfile)

    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print("Loss (avg)",metric_dict['loss_mean']/folds, "Acc (avg)",metric_dict['acc_mean'],'F1 (avg)',metric_dict['f1_mean'],'AUCROC (avg)',metric_dict['aucroc_mean'],'AUCPR',metric_dict['aucpr_mean'])
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')