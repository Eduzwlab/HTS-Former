import numpy as np
from sklearn.metrics import  auc, roc_curve, roc_auc_score, accuracy_score, confusion_matrix,precision_score,recall_score,f1_score,classification_report
import torch
import matplotlib.pyplot as plt
from itertools import cycle
import seaborn as sns

def seed_torch(seed=2021):
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False   

@torch.no_grad()
def ema_update(model,targ_model,mm=0.9999):
    r"""Performs a momentum update of the target network's weights.
    Args:
        mm (float): Momentum used in moving average update.
    """
    assert 0.0 <= mm <= 1.0, "Momentum needs to be between 0.0 and 1.0, got %.5f" % mm

    for param_q, param_k in zip(model.parameters(), targ_model.parameters()):
        param_k.data.mul_(mm).add_(param_q.data, alpha=1. - mm) # mm*k +(1-mm)*q

def patch_shuffle(x,group=0,g_idx=None,return_g_idx=False):
    b,p,n = x.size()
    ps = torch.tensor(list(range(p)))

    # padding
    H, W = int(np.ceil(np.sqrt(p))), int(np.ceil(np.sqrt(p)))
    if group > H or group<= 0:
        return group_shuffle(x,group)
    _n = -H % group
    H, W = H+_n, W+_n
    add_length = H * W - p
    ps = torch.cat([ps,torch.tensor([-1 for i in range(add_length)])])
    # patchify
    ps = ps.reshape(shape=(group,H//group,group,W//group))
    ps = torch.einsum('hpwq->hwpq',ps)
    ps = ps.reshape(shape=(group**2,H//group,W//group))
    # shuffle
    if g_idx is None:
        g_idx = torch.randperm(ps.size(0))
    ps = ps[g_idx]
    # unpatchify
    ps = ps.reshape(shape=(group,group,H//group,W//group))
    ps = torch.einsum('hwpq->hpwq',ps)
    ps = ps.reshape(shape=(H,W))
    idx = ps[ps>=0].view(p)
    
    if return_g_idx:
        return x[:,idx.long()],g_idx
    else:
        return x[:,idx.long()]

def group_shuffle(x,group=0):
    b,p,n = x.size()
    ps = torch.tensor(list(range(p)))
    if group > 0 and group < p:
        _pad = -p % group
        ps = torch.cat([ps,torch.tensor([-1 for i in range(_pad)])])
        ps = ps.view(group,-1)
        g_idx = torch.randperm(ps.size(0))
        ps = ps[g_idx]
        idx = ps[ps>=0].view(p)
    else:
        idx = torch.randperm(p)
    return x[:,idx.long()]


def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]



def make_weights_for_balanced_classes_split(dataset):
    N = float(len(dataset))
    labels = np.array(dataset.slide_label)
    label_uni = set(dataset.slide_label)
    weight_per_class = [N/len(labels[labels==c]) for c in label_uni]
    weight = [0] * int(N)
    for idx in range(len(dataset)):
        y = dataset.slide_label[idx]
        weight[idx] = weight_per_class[y]

    return torch.DoubleTensor(weight)

def plot_roc_curve_multiclass(y_true, y_prob, num_classes):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve((y_true == i), y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8, 6))

    colors = ['aqua', 'darkorange', 'cornflowerblue']
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'Class {i} (AUC = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve for Multi-class')
    plt.legend(loc='lower right')
    plt.show()


def five_scores(bag_labels, bag_predictions):

    auc_value = roc_auc_score(bag_labels, bag_predictions, multi_class='ovr', average='micro')
    auc_value_macro = roc_auc_score(bag_labels, bag_predictions, multi_class='ovr', average='macro')
    auc_value_weighted = roc_auc_score(bag_labels, bag_predictions, multi_class='ovr', average='weighted')

    y_pred = np.argmax(bag_predictions, axis=1)
    y_pred =y_pred.tolist()

    accuracy = accuracy_score(bag_labels, y_pred)
    target_names = ['Basal 0', 'LumA 1', 'LumB 2']
    three_scores = classification_report(bag_labels, y_pred, target_names=target_names)
    print(classification_report(bag_labels, y_pred, target_names=target_names))

    # Precision
    precision = precision_score(bag_labels, y_pred, average='micro')
    precision_macro = precision_score(bag_labels, y_pred, average='macro')
    precision_weighted = precision_score(bag_labels, y_pred, average='weighted')


    # Recall
    recall = recall_score(bag_labels, y_pred, average='micro')
    recall_macro = recall_score(bag_labels, y_pred, average='macro')
    recall_weighted = recall_score(bag_labels, y_pred, average='weighted')

    #F1-score
    fscore = f1_score(bag_labels, y_pred, average='micro')
    fscore_macro = f1_score(bag_labels, y_pred, average='macro')
    fscore_weighted = f1_score(bag_labels, y_pred, average='weighted')

    return accuracy, auc_value,auc_value_macro,auc_value_weighted, precision,precision_macro,precision_weighted, recall,recall_macro,recall_weighted, fscore,fscore_macro,fscore_weighted,three_scores



def five_scores_final(bag_labels, bag_predictions):
    auc_value = roc_auc_score(bag_labels, bag_predictions, multi_class='ovr', average='micro')
    auc_value_macro = roc_auc_score(bag_labels, bag_predictions, multi_class='ovr', average='macro')
    auc_value_weighted = roc_auc_score(bag_labels, bag_predictions, multi_class='ovr', average='weighted')

    y_pred = np.argmax(bag_predictions, axis=1)
    y_pred =y_pred.tolist()

    accuracy = accuracy_score(bag_labels, y_pred)
    target_names = ['Basal 0', 'LumA 1', 'LumB 2']
    three_scores = classification_report(bag_labels, y_pred, target_names=target_names)
    print(classification_report(bag_labels, y_pred, target_names=target_names))

    # Precision
    precision = precision_score(bag_labels, y_pred, average='micro')
    precision_macro = precision_score(bag_labels, y_pred, average='macro')
    precision_weighted = precision_score(bag_labels, y_pred, average='weighted')

    # Recall
    recall = recall_score(bag_labels, y_pred, average='micro')
    recall_macro = recall_score(bag_labels, y_pred, average='macro')
    recall_weighted = recall_score(bag_labels, y_pred, average='weighted')

    # F1-score
    fscore = f1_score(bag_labels, y_pred, average='micro')
    fscore_macro = f1_score(bag_labels, y_pred, average='macro')
    fscore_weighted = f1_score(bag_labels, y_pred, average='weighted')

    return accuracy, auc_value,auc_value_macro,auc_value_weighted, precision,precision_macro,precision_weighted, recall,recall_macro,recall_weighted, fscore,fscore_macro,fscore_weighted,three_scores,bag_labels, bag_predictions,y_pred


def plot_roc_curve(y_true, y_prob, num_classes,model_path):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_true = np.array(y_true)


    # Compute ROC curve and AUC for each class
    for i in range(num_classes):
        class_prob_data = [prob[i] for prob in y_prob]
        list_temp = np.array(class_prob_data)
        fpr[i], tpr[i], _ = roc_curve((y_true == i), list_temp)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and AUC
    y_true_onehot = np.eye(num_classes)[y_true]  # One-hot encode the true labels
    y_prob = np.array(y_prob)
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_onehot.ravel(), y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and AUC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= num_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    lw = 2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    class_name = ["Basal","LumA","LumB"]
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(class_name[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('multi-calss ROC')
    plt.legend(loc="lower right")
    # plt.savefig('{}/{}_{}.png'.format(resultsPath, "ROC", key))
    plt.savefig('{}/{}.png'.format(model_path, "roc_curve"))
    plt.show()

def plot_confusion_matrix(y_true, y_pred, num_classes,model_path):

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    class_name = ["Basal", "LumA", "LumB"]
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(num_classes), yticklabels=np.arange(num_classes))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_name, yticklabels=class_name)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix for Multi-class')
    plt.savefig('{}/{}.png'.format(model_path, "Confusion_Matrix"))
    plt.show()


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))


    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False,save_best_model_stage=0.):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.save_best_model_stage = save_best_model_stage

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):
        
        score = -val_loss if epoch >= self.save_best_model_stage else 0.

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def state_dict(self):
        return {
            'patience': self.patience,
            'stop_epoch': self.stop_epoch,
            'verbose': self.verbose,
            'counter': self.counter,
            'best_score': self.best_score,
            'early_stop': self.early_stop,
            'val_loss_min': self.val_loss_min
        }
    def load_state_dict(self,dict):
        self.patience = dict['patience']
        self.stop_epoch = dict['stop_epoch']
        self.verbose = dict['verbose']
        self.counter = dict['counter']
        self.best_score = dict['best_score']
        self.early_stop = dict['early_stop']
        self.val_loss_min = dict['val_loss_min']

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        #torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss
