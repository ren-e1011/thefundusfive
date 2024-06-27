
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
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, average_precision_score, multilabel_confusion_matrix, confusion_matrix, ConfusionMatrixDisplay
from timm.utils import accuracy
import matplotlib.pyplot as plt

import torch.nn.functional as F
from torch.nn import Softmax
from RETFound_MAE.util.misc import MetricLogger

from utils import misc_measures
import pickle
import wandb

import warnings 


def evaluate_metrics(images, target, model, criterion, mode = 'val', n_classes = 4, metric_logger = None):
    # self.target_dict = {'A':0,'D':1,'G':2,'N':3} where A is AMD and the rest should be 'other' 
    if mode == 'test':
        target_binmod = {0:1,1:0,2:0,3:0}
        target = torch.tensor([target_binmod[t.item()] for t in target])

    true_label=F.one_hot(target.to(torch.int64), num_classes=n_classes)

    # compute output
    # with torch.autocast(device_type=device):
    output = model(images)
    loss = criterion(output, target)
    prediction_softmax = Softmax(dim=1)(output)
    _,prediction_decode = torch.max(prediction_softmax, 1)

    prediction_decode = torch.tensor([target_binmod[t.item()] for t in prediction_decode]) # y_preds

    _,true_label_decode = torch.max(true_label, 1) # same as target ie y

    acc1,acc2 = accuracy(output, target, topk=(1,2)) # leave topk as 1 mod from topk=(1,2)


    if metric_logger: 
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=images.shape[0])

    return prediction_decode, true_label_decode, true_label, prediction_softmax, loss

# slight mods from RETFound_MAE engine_finetune evaluate
@torch.no_grad()
def evaluate(data_loader, model, logger = None, k='', mode='val', n_classes=4):
    
    criterion = torch.nn.CrossEntropyLoss()
    # switch to evaluation mode
    model.eval()
    

    if mode == 'val':

        metric_logger = MetricLogger(delimiter="  ")
        header = 'Test:'

        prediction_decode_list = []
        prediction_list = []
        true_label_decode_list = []
        true_label_onehot_list = []
    
        for batch in metric_logger.log_every(data_loader, 10, header):
            images = batch[0]
            target = batch[-1]

            prediction_decode, true_label_decode, true_label, prediction_softmax, loss = evaluate_metrics(images,target,model,criterion, mode,n_classes)

            prediction_decode_list.extend(prediction_decode.cpu().detach().numpy())
            true_label_decode_list.extend(true_label_decode.cpu().detach().numpy())
            true_label_onehot_list.extend(true_label.cpu().detach().numpy())
            prediction_list.extend(prediction_softmax.cpu().detach().numpy())
        

        auc_roc = roc_auc_score(true_label_onehot_list, prediction_list,multi_class='ovr',average='macro')
        auc_pr = average_precision_score(true_label_onehot_list, prediction_list,average='macro')  

        true_label_decode_list = np.array(true_label_decode_list)
        prediction_decode_list = np.array(prediction_decode_list)
        conf_mat = multilabel_confusion_matrix(true_label_decode_list, prediction_decode_list,labels=[i for i in range(n_classes)])
        acc, sensitivity, specificity, precision, G, F1, mcc = misc_measures(conf_mat)        
                
        
        print('Sklearn Metrics - Acc: {:.4f} AUC-roc: {:.4f} AUC-pr: {:.4f} F1-score: {:.4f} MCC: {:.4f}'.format(acc, auc_roc, auc_pr, F1, mcc)) 
        
        # wandb logging 
        if logger: 
            logger.log({f"{mode}_loss":loss, f"{mode}_accuracy":acc, 'F1-score':F1, 'AUC_ROC':auc_roc, 'AUC_Precision':auc_pr})
        else:
            print(f"{mode}_loss:",loss, f"{mode}_accuracy:",acc, 'F1-score:',F1, 'AUC_ROC:',auc_roc, 'AUC_Precision:',auc_pr)
    
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()},auc_roc
    
    elif mode == 'test':
        images, target = data_loader

        y_pred, y, y_oh , _, loss = evaluate_metrics(images,target,model,criterion, mode, n_classes=2)
        auc_roc = roc_auc_score(y, y_pred,multi_class='ovr',average='macro')
        conf_mat =  confusion_matrix(y,y_pred)   # vs true_label? 
        cmd = ConfusionMatrixDisplay(conf_mat,display_labels={1:'AMD',0:'Other'})
        cmd.plot()
        # cmdp = cmd.plot(cmap=plt.cm.Blues,plot_lib="matplotlib")
        plt.savefig(f"fold_{k}_confusion_matrix_test.jpg",bbox_inches ='tight')
        acc = accuracy_score(y,y_pred)
        
        F1 = f1_score(y,y_pred)

        return acc, F1, auc_roc, conf_mat
        # gather the stats from all processes
        
        
    # TypeError: 'numpy.ndarray' object is not callable
    # if mode=='test':
    #     cm = confusion_matrix(actual_vector=true_label_decode_list, predict_vector=prediction_decode_list)
    #     cm.plot(cmap=plt.cm.Blues,number_label=True,normalized=True,plot_lib="matplotlib")
    #     plt.savefig(task+'confusion_matrix_test.jpg',dpi=600,bbox_inches ='tight')
    
    else:
        warnings.warn(f"Evaluate accepts 'val' and 'test' modes. Your input mode is {mode}")
        return 

def test_retfund_fives(dataset, testing_params, data_params):
    test_ids = [i for i in range(len(dataset))]
    test_sampler = SubsetRandomSampler(test_ids)
    n_classes = testing_params.n_classes # 2
    folds = testing_params.folds 

    data_loader_test = DataLoader(
        dataset, sampler=test_sampler,
        batch_size= testing_params.batch_size
    )

    for x, y in data_loader_test:
        break

    acc_mean = 0.
    f1_mean = 0.
    auc_roc_mean = 0.

    for k in range(folds):
        print('~~~~~~~~')
        print(f"Fold {k}")
        print('~~~~~~~~')
        mdl_at = f"{testing_params.task}_fold_{k}_ckpt-best.pth"

        model = models_vit.__dict__['vit_large_patch16'](
            img_size = data_params.out_size, # config.data.input_size 
            num_classes=4, # load with 4 to prevent mismat
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


        acc, F1, auc_roc, conf_mat = evaluate((x,y),model,k=str(k), mode='test',n_classes=n_classes) 

        acc_mean += acc
        f1_mean += F1
        auc_roc_mean += auc_roc

        print('Test accuracy', acc, 'Test F1', F1, 'Test AUC ROC', auc_roc)
        print('Confusion Matrix')
        print(conf_mat)

    acc_mean /= folds
    f1_mean /= folds
    auc_roc_mean /= folds

    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print( "Acc (avg)", acc_mean ,'F1 (avg)', f1_mean,'AUCROC (avg)',auc_roc_mean)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
