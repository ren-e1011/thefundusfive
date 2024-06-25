import math
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, average_precision_score,multilabel_confusion_matrix

from torch.nn import Softmax
from torch import max 

def accuracy(outputs, y):
    preds, yhat = max(Softmax(dim=1)(outputs),1)
    return accuracy_score(y,yhat)

def configure_optimizer( model,data_config, train_config, optimizer = None):
        lr = train_config.lr
        weight_decay = train_config.weight_decay
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay) if not optimizer else optimizer

        nsteps_train = int(data_config.nsamples * data_config.train_eval_split / train_config.batch_size)

        scheduler_params = train_config.lr_scheduler
        if not scheduler_params.use:
            return optimizer

        # GC
        # total_steps = scheduler_params.total_steps
        # assert total_steps is not None
        # assert total_steps > 0

        # Here we interpret the final lr as max_lr/final_div_factor.
        # Note that Pytorch OneCycleLR interprets it as initial_lr/final_div_factor:
        final_div_factor_pytorch = scheduler_params.final_div_factor / scheduler_params.div_factor
        lr_scheduler = OneCycleLR(
            optimizer=optimizer,
            max_lr=lr,
            steps_per_epoch= nsteps_train,
            epochs= train_config.exp_epochs * 2,
            div_factor=scheduler_params.div_factor,
            final_div_factor=final_div_factor_pytorch,
            # total_steps=total_steps, instead, calculate steps from expected 25 epch and steps per epch
            # pct_start=scheduler_params.pct_start, instead take dfault percentage of cycle increasing lr 
            cycle_momentum=False,
            anneal_strategy='cos') # mod from linear
        # lr_scheduler_config = {
        #     "scheduler": lr_scheduler,
        #     "interval": "step",
        #     "frequency": 1,
        #     "strict": True,
        #     "name": 'learning_rate',
        # }

        # return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}
        return optimizer, lr_scheduler

# slight mod from RETFound_MAE.uti
def adjust_learning_rate(optimizer, epoch, lr_params):

    lr = lr_params.lr
    min_lr = lr_params.min_lr
    epochs = lr_params.max_epch
    warmup_epochs = lr_params.warmup_epochs

    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < warmup_epochs:
        lr = lr_params.lr
        lr = lr * epoch / warmup_epochs 
    else:
        lr = min_lr + (lr - min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

# lifted verbatim from RETFound_MAE engine_finetune misc_measures 
def misc_measures(confusion_matrix):
    
    acc = []
    sensitivity = []
    specificity = []
    precision = []
    G = []
    F1_score_2 = []
    mcc_ = []
    
    for i in range(1, confusion_matrix.shape[0]):
        cm1=confusion_matrix[i]
        acc.append(1.*(cm1[0,0]+cm1[1,1])/np.sum(cm1))
        sensitivity_ = 1.*cm1[1,1]/(cm1[1,0]+cm1[1,1])
        sensitivity.append(sensitivity_)
        specificity_ = 1.*cm1[0,0]/(cm1[0,1]+cm1[0,0])
        specificity.append(specificity_)
        precision_ = 1.*cm1[1,1]/(cm1[1,1]+cm1[0,1])
        precision.append(precision_)
        G.append(np.sqrt(sensitivity_*specificity_))
        F1_score_2.append(2*precision_*sensitivity_/(precision_+sensitivity_))
        mcc = (cm1[0,0]*cm1[1,1]-cm1[0,1]*cm1[1,0])/np.sqrt((cm1[0,0]+cm1[0,1])*(cm1[0,0]+cm1[1,0])*(cm1[1,1]+cm1[1,0])*(cm1[1,1]+cm1[0,1]))
        mcc_.append(mcc)
        
    acc = np.array(acc).mean()
    sensitivity = np.array(sensitivity).mean()
    specificity = np.array(specificity).mean()
    precision = np.array(precision).mean()
    G = np.array(G).mean()
    F1_score_2 = np.array(F1_score_2).mean()
    mcc_ = np.array(mcc_).mean()
    
    return acc, sensitivity, specificity, precision, G, F1_score_2, mcc_