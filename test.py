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

# slight mods from RETFound_MAE engine_finetune evaluate
@torch.no_grad()
def evaluate(data_loader, model, device, out_dir, epoch, k='', mode='val', n_classes=4):
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

        acc1,_ = accuracy(output, target, topk=(1,2))

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
    results_path = task+'_metrics_{}.csv'.format(mode)
    with open(results_path,mode='a',newline='',encoding='utf8') as cfa:
        wf = csv.writer(cfa)
        data2=[[acc,sensitivity,specificity,precision,auc_roc,auc_pr,F1,mcc,metric_logger.loss]]
        for i in data2:
            wf.writerow(i)
            
    
    if mode=='test':
        cm = confusion_matrix(actual_vector=true_label_decode_list, predict_vector=prediction_decode_list)
        cm.plot(cmap=plt.cm.Blues,number_label=True,normalized=True,plot_lib="matplotlib")
        plt.savefig(task+'confusion_matrix_test.jpg',dpi=600,bbox_inches ='tight')
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()},auc_roc


def test_retfund_fives(dataset, batch_size):
    test_ids = [i for i in len(dataset)]
    test_sampler = SubsetRandomSampler(test_ids)

    data_loader_test = DataLoader(
        dataset, sampler=test_sampler,
        batch_size= batch_size
    )

    # load model similar to train TODO 

    # state_dict_best = torch.load(task+'checkpoint-best.pth', map_location='cpu')
    # model.load_state_dict(state_dict_best['model'])
    # test_stats,auc_roc = evaluate(data_loader_test, model, device, task,epoch=0, mode='test',num_class=n_classes)

