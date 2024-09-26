import wandb
import torch.nn as nn
from dataloader import *
from torch.utils.data import DataLoader, RandomSampler
import argparse
from modules import attmil,clam, htsfomer, dsmil,transmil,mean_max
from modules.MambaMIL import MambaMIL
from modules.S4MIL.s4model import S4Model
from torch.nn.functional import one_hot
from torch.cuda.amp import GradScaler
from contextlib import suppress
from modules.abmil import AB_MIL_Attention
import time
from timm.utils import AverageMeter,dispatch_clip_grad
from timm.models import  model_parameters
from collections import OrderedDict
from utils import *
import pandas as pd
import os

def main(args):
    seed_torch(args.seed)
    if args.datasets.lower() == 'tcga':
        label_path=os.path.join(args.dataset_root,'label.csv')
        p, l = get_patient_label(label_path)
        index = [i for i in range(len(p))]
        random.shuffle(index)
        p = p[index]
        l = l[index]

    if args.cv_fold > 1:
        train_p, train_l, test_p, test_l,val_p,val_l = get_kflod(args.cv_fold, p, l,args.val_ratio)
        history = pd.DataFrame(list(zip(train_p, train_l, val_p,val_l,test_p, test_l)),
                               columns=['train_p', 'train_l', 'val_p', 'val_l','test_p','test_l'])
        history.to_csv(os.path.join(args.model_path, 'split' + '.csv'), index=False)

    acs, pre, pre_mac, pre_wei, rec, rec_mac, rec_wei, fs, fs_mac, fs_wei, auc, auc_mac, auc_wei, te_auc, te_fs,bag_l, bag_pre, p_name,pred_int = [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
    ckc_metric = [acs, pre, pre_mac, pre_wei, rec, rec_mac, rec_wei, fs, fs_mac, fs_wei, auc, auc_mac, auc_wei, te_auc, te_fs,bag_l, bag_pre, p_name,pred_int]

    if not args.no_log:
        print('Dataset: ' + args.datasets)

    # resume
    if args.auto_resume and not args.no_log:
        ckp = torch.load(os.path.join(args.model_path,'ckp.pt'))
        args.fold_start = ckp['k']
        if len(ckp['ckc_metric']) == 6:
            acs, pre, rec,fs,auc,te_auc = ckp['ckc_metric']
        elif len(ckp['ckc_metric']) == 7:
            acs, pre, rec,fs,auc,te_auc,te_fs = ckp['ckc_metric']
        else:
            acs, pre, rec,fs,auc = ckp['ckc_metric']

    for k in range(args.fold_start, args.cv_fold):
        if not args.no_log:
            print('Start %d-fold cross validation: fold %d ' % (args.cv_fold, k))
        ckc_metric = one_fold(args,k,ckc_metric,train_p, train_l, test_p, test_l,val_p,val_l)

    bag_pre_merged_list = [element for sublist in bag_pre for element in sublist]
    p_name_merged_list = [element for sublist in p_name for element in sublist]
    bag_l_merged_list = [element for sublist in bag_l for element in sublist]
    pred_int_merged_list = [element for sublist in pred_int for element in sublist]

    datasave = {
        "Name": p_name_merged_list,
        "True Label": bag_l_merged_list,
        "Prediction": pred_int_merged_list,
    }
    for i in range(len(bag_pre_merged_list[0])):

        datasave[f'Probability Class {i}'] = [prob[i] for prob in bag_pre_merged_list]

    df = pd.DataFrame(datasave)
    df.to_csv(os.path.join(args.model_path, 'TEST_RESULT_PATIENT_BASED_FINAL' + '.csv'), index=False)

    plot_roc_curve(bag_l_merged_list, bag_pre_merged_list, args.n_classes,args.model_path)
    plot_confusion_matrix(bag_l_merged_list, pred_int_merged_list, args.n_classes,args.model_path)

    if args.always_test:
        if args.wandb:
            wandb.log({
                "cross_val/te_auc_mean":np.mean(np.array(te_auc)),
                "cross_val/te_auc_std":np.std(np.array(te_auc)),
                "cross_val/te_f1_mean":np.mean(np.array(te_fs)),
                "cross_val/te_f1_std":np.std(np.array(te_fs)),
            })

    if args.wandb:
        wandb.log({
            "cross_val/acc_mean":np.mean(np.array(acs)),
            "cross_val/auc_mean":np.mean(np.array(auc)),
            "cross_val/auc_mac_mean":np.mean(np.array(auc_mac)),
            "cross_val/auc_wei_mean":np.mean(np.array(auc_wei)),
            "cross_val/f1_mean":np.mean(np.array(fs)),
            "cross_val/f1_mac_mean":np.mean(np.array(fs_mac)),
            "cross_val/f1_wei_mean":np.mean(np.array(auc_wei)),
            "cross_val/pre_mean":np.mean(np.array(pre)),
            "cross_val/pre_mac_mean":np.mean(np.array(pre_mac)),
            "cross_val/pre_wei_mean":np.mean(np.array(pre_wei)),
            "cross_val/recall_mean":np.mean(np.array(rec)),
            "cross_val/recall_mac_mean":np.mean(np.array(rec_mac)),
            "cross_val/recall_wei_mean":np.mean(np.array(rec_wei)),
            "cross_val/acc_std":np.std(np.array(acs)),
            "cross_val/auc_std":np.std(np.array(auc)),
            "cross_val/auc_mac_std":np.std(np.array(auc_mac)),
            "cross_val/auc_wei_std":np.std(np.array(auc_wei)),
            "cross_val/f1_std":np.std(np.array(fs)),
            "cross_val/f1_mac_std":np.std(np.array(fs_mac)),
            "cross_val/f1_wei_std":np.std(np.array(fs_wei)),
            "cross_val/pre_std":np.std(np.array(pre)),
            "cross_val/pre_mac_std":np.std(np.array(pre_mac)),
            "cross_val/pre_wei_std":np.std(np.array(pre_wei)),
            "cross_val/recall_std":np.std(np.array(rec)),
            "cross_val/recall_mac_std":np.std(np.array(rec_mac)),
            "cross_val/recall_wei_std":np.std(np.array(rec_wei)),
        })
    if not args.no_log:
        print('Cross validation accuracy mean: %.3f, std %.3f ' % (np.mean(np.array(acs)), np.std(np.array(acs))))
        print('Cross validation auc mean: %.3f, std %.3f ' % (np.mean(np.array(auc)), np.std(np.array(auc))))
        print('Cross validation auc_mac mean: %.3f, std %.3f ' % (np.mean(np.array(auc_mac)), np.std(np.array(auc_mac))))
        print('Cross validation auc_wei mean: %.3f, std %.3f ' % (np.mean(np.array(auc_wei)), np.std(np.array(auc_wei))))
        print('Cross validation precision mean: %.3f, std %.3f ' % (np.mean(np.array(pre)), np.std(np.array(pre))))
        print('Cross validation precision_mac mean: %.3f, std %.3f ' % (np.mean(np.array(pre_mac)), np.std(np.array(pre_mac))))
        print('Cross validation precision_wei mean: %.3f, std %.3f ' % (np.mean(np.array(pre_wei)), np.std(np.array(pre_wei))))
        print('Cross validation recall mean: %.3f, std %.3f ' % (np.mean(np.array(rec)), np.std(np.array(rec))))
        print('Cross validation recall_mac mean: %.3f, std %.3f ' % (np.mean(np.array(rec_mac)), np.std(np.array(rec_mac))))
        print('Cross validation recall_wei mean: %.3f, std %.3f ' % (np.mean(np.array(rec_wei)), np.std(np.array(rec_wei))))
        print('Cross validation fscore mean: %.3f, std %.3f ' % (np.mean(np.array(fs)), np.std(np.array(fs))))
        print('Cross validation fscore_mac mean: %.3f, std %.3f ' % (np.mean(np.array(fs_mac)), np.std(np.array(fs_mac))))
        print('Cross validation fscore_wei mean: %.3f, std %.3f ' % (np.mean(np.array(fs_wei)), np.std(np.array(fs_wei))))

    # f.close()

def one_fold(args,k,ckc_metric,train_p, train_l, test_p, test_l,val_p,val_l):
    # --->initiation
    seed_torch(args.seed)
    loss_scaler = GradScaler() if args.amp else None
    amp_autocast = torch.cuda.amp.autocast if args.amp else suppress
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    acs, pre, pre_mac, pre_wei, rec, rec_mac, rec_wei, fs, fs_mac, fs_wei, auc, auc_mac, auc_wei, te_auc, te_fs,bag_l, bag_pre, p_name,pred_int = ckc_metric

    # --->load data
    if args.datasets.lower() == 'tcga':
        
        train_set = TCGADataset(train_p[k],train_l[k],args.tcga_max_patch,args.dataset_root,persistence=args.persistence,keep_same_psize=args.same_psize,is_train=True)
        test_set = TCGADataset(test_p[k],test_l[k],args.tcga_max_patch,args.dataset_root,persistence=args.persistence,keep_same_psize=args.same_psize)
        if args.val_ratio != 0.:
            val_set = TCGADataset(val_p[k],val_l[k],args.tcga_max_patch,args.dataset_root,persistence=args.persistence,keep_same_psize=args.same_psize)
        else:
            val_set = test_set

    if args.fix_loader_random:
        # generated by int(torch.empty((), dtype=torch.int64).random_().item())
        big_seed_list = 7784414403328510413
        generator = torch.Generator()
        generator.manual_seed(big_seed_list)  
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,generator=generator)
    else:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=RandomSampler(train_set), num_workers=args.num_workers)

    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    mm_sche = None

    # --->bulid networks
    if args.model == 'hts-fomer':
        if args.mrh_sche:
            mrh_sche = cosine_scheduler(args.mask_ratio_h,0.,epochs=args.num_epoch,niter_per_ep=len(train_loader))
        else:
            mrh_sche = None

        model_params = {
            'baseline': args.baseline,
            'dropout': args.dropout,
            'mask_ratio' : args.mask_ratio,
            'n_classes': args.n_classes,
            'temp_t': args.temp_t,
            'act': args.act,
            'head': args.n_heads,
            'msa_fusion': args.msa_fusion,
            'mask_ratio_h': args.mask_ratio_h,
            'mask_ratio_hr': args.mask_ratio_hr,
            'mask_ratio_l': args.mask_ratio_l,
            'mrh_sche': mrh_sche,
            'da_act': args.da_act,
            'attn_layer': args.attn_layer,
        }
        
        if args.mm_sche:
            mm_sche = cosine_scheduler(args.mm,args.mm_final,epochs=args.num_epoch,niter_per_ep=len(train_loader),start_warmup_value=1.)

        model = htsfomer.Former(**model_params).to(device)
    elif args.model == 'attmil':
        model = attmil.DAttention(n_classes=args.n_classes,dropout=args.dropout,act=args.act).to(device)
    elif args.model == 'gattmil':
        model = attmil.AttentionGated(dropout=args.dropout).to(device)
    elif args.model == 'abmil':
        model = AB_MIL_Attention(in_size=args.num_feats_ab, out_size=args.n_classes, confounder_path=None, confounder_learn='store_true', confounder_dim=128,
                                 confounder_merge='cat').to(device)
    elif args.model == 'mambamil':
        model = MambaMIL(in_dim=args.num_feats_ab, n_classes=args.n_classes, dropout=args.dropout, act='gelu',
                     layer=args.mambamil_layer, rate=args.mambamil_rate, type=args.mambamil_type).to(device)
    elif args.model == 's4mil':
        model = S4Model(n_class=args.n_classes).to(device)

    elif args.model == 'clam_sb':
        model = clam.CLAM_SB(n_classes=args.n_classes,dropout=args.dropout,act=args.act).to(device)
    elif args.model == 'clam_mb':
        model = clam.CLAM_MB(n_classes=args.n_classes,dropout=args.dropout,act=args.act).to(device)
    elif args.model == 'transmil':
        model = transmil.TransMIL(n_classes=args.n_classes,dropout=args.dropout,act=args.act).to(device)
    elif args.model == 'dsmil':
        i_classifier = dsmil.FCLayer(in_size=args.num_feats, out_size=args.n_classes)
        b_classifier = dsmil.BClassifier(input_size=args.num_feats, output_class=args.n_classes)
        model = dsmil.MILNet(i_classifier, b_classifier).to(device)

    elif args.model == 'meanmil':
        model = mean_max.MeanMIL(n_classes=args.n_classes,dropout=args.dropout,act=args.act).to(device)
    elif args.model == 'maxmil':
        model = mean_max.MaxMIL(n_classes=args.n_classes,dropout=args.dropout,act=args.act).to(device)




    if args.loss == 'bce':
        criterion = nn.BCEWithLogitsLoss()
        # criterion = nn.BCELoss()
    elif args.loss == 'ce':
        criterion = nn.CrossEntropyLoss()

    # optimizer
    if args.opt == 'adamw':
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    if args.lr_sche == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epoch, 0) if not args.lr_supi else torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epoch*len(train_loader), 0)
    elif args.lr_sche == 'step':
        assert not args.lr_supi
        # follow the DTFD-MIL
        # ref:https://github.com/hrzhang1123/DTFD-MIL
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,args.num_epoch / 2, 0.2)
    elif args.lr_sche == 'const':
        scheduler = None

    if args.early_stopping:
        early_stopping = EarlyStopping(patience= 20, stop_epoch= 70,save_best_model_stage=np.ceil(args.save_best_model_stage * args.num_epoch))
    else:
        early_stopping = None

    optimal_ac, opt_pre, opt_re, opt_fs, opt_auc,opt_epoch = 0, 0, 0, 0,0,0
    epoch_start = 0

    if args.fix_train_random:
        seed_torch(args.seed)

    # resume
    if args.auto_resume and not args.no_log:
        ckp = torch.load(os.path.join(args.model_path,'ckp.pt'))
        epoch_start = ckp['epoch']
        model.load_state_dict(ckp['model'])
        optimizer.load_state_dict(ckp['optimizer'])
        scheduler.load_state_dict(ckp['lr_sche'])
        early_stopping.load_state_dict(ckp['early_stop'])
        optimal_ac, opt_pre, opt_re, opt_fs, opt_auc,opt_epoch = ckp['val_best_metric']
        opt_te_auc = ckp['te_best_metric'][0]
        if len(ckp['te_best_metric']) > 1:
            opt_te_fs = ckp['te_best_metric'][1]
        np.random.set_state(ckp['random']['np'])
        torch.random.set_rng_state(ckp['random']['torch'])
        random.setstate(ckp['random']['py'])
        if args.fix_loader_random:
            train_loader.sampler.generator.set_state(ckp['random']['loader'])
        args.auto_resume = False

    train_time_meter = AverageMeter()

    for epoch in range(epoch_start, args.num_epoch):
        train_loss,start,end = train_loop(args,model,train_loader,optimizer,device,amp_autocast,criterion,loss_scaler,scheduler,k,mm_sche,epoch)
        train_time_meter.update(end-start)
        stop,accuracy, auc_value, auc_value_macro, auc_value_weighted, precision, precision_macro, precision_weighted, recall, recall_macro, recall_weighted, fscore, fscore_macro, fscore_weighted, three_scores, test_loss = val_loop(args,model,val_loader,device,criterion,early_stopping,epoch)



        if args.always_test:
            _te_accuracy, _te_auc_value, _te_auc_value_macro, _te_auc_value_weighted, _te_precision, _te_precision_macro, _te_precision_weighted, _te_recall, _te_recall_macro, _te_recall_weighted, _te_fscore, _te_fscore_macro, _te_fscore_weighted, _te_three_scores,_te_test_loss_log = test(args,model,test_loader,device,criterion)
            if args.wandb:
                rowd = OrderedDict([
                    ("te_acc",_te_accuracy),
                    ("te_precision",_te_precision),
                    ("te_precision_macro",_te_precision_macro),
                    ("te_precision_weighted",_te_precision_weighted),
                    ("te_recall",_te_recall),
                    ("te_recall_macro",_te_recall_macro),
                    ("te_recall_weighted",_te_recall_weighted),
                    ("te_fscore",_te_fscore),
                    ("te_fscore_macro",_te_fscore_macro),
                    ("te_fscore_weighted",_te_fscore_weighted),
                    ("te_auc",_te_auc_value),
                    ("te_auc_macro",_te_auc_value_macro),
                    ("te_auc_weighted",_te_auc_value_weighted),
                    ("te_loss",_te_test_loss_log),
                    ("te_three_scores",_te_three_scores),
                ])

                rowd = OrderedDict([ (str(k)+'-fold/'+_k,_v) for _k, _v in rowd.items()])
                wandb.log(rowd)

            if _te_auc_value > opt_te_auc:
                opt_te_auc = _te_auc_value
                opt_te_fs = _te_fscore
                if args.wandb:
                    rowd = OrderedDict([
                        ("best_te_auc",opt_te_auc),
                        ("best_te_f1",_te_fscore)
                    ])
                    rowd = OrderedDict([ (str(k)+'-fold/'+_k,_v) for _k, _v in rowd.items()])
                    wandb.log(rowd)


        if not args.no_log:
            print('\r Epoch [%d/%d] train loss: %.1E, test loss: %.1E, accuracy: %.3f, auc_value:%.3f, precision: %.3f, recall: %.3f, fscore: %.3f , time: %.3f(%.3f)' %
        (epoch+1, args.num_epoch, train_loss, test_loss, accuracy, auc_value, precision, recall, fscore, train_time_meter.val,train_time_meter.avg))

        if args.wandb:
            rowd = OrderedDict([
                ("val_acc",accuracy),
                ("val_precision",precision),
                ("val_precision_macro",precision_macro),
                ("val_precision_weighted",precision_weighted),
                ("val_recall",recall),
                ("val_recall_macro",recall_macro),
                ("val_recall_weighted",recall_weighted),
                ("val_fscore",fscore),
                ("val_fscore_macro",fscore_macro),
                ("val_fscore_weighted",fscore_weighted),
                ("val_auc",auc_value),
                ("val_auc_macro",auc_value_macro),
                ("val_auc_weighted",auc_value_weighted),
                ("val_loss",test_loss),
                ("three_scores",three_scores),
                ("epoch",epoch),
            ])

            rowd = OrderedDict([ (str(k)+'-fold/'+_k,_v) for _k, _v in rowd.items()])
            wandb.log(rowd)

        if auc_value > opt_auc and epoch >= args.save_best_model_stage*args.num_epoch:
            optimal_ac = accuracy
            opt_pre = precision
            opt_re = recall
            opt_fs = fscore
            opt_auc = auc_value
            opt_epoch = epoch

            if not os.path.exists(args.model_path):
                os.mkdir(args.model_path)
            if not args.no_log:
                best_pt = {
                    'model': model.state_dict(),
                }
                torch.save(best_pt, os.path.join(args.model_path, 'fold_{fold}_model_best_auc.pt'.format(fold=k)))
        if args.wandb:
            rowd = OrderedDict([
                ("val_best_acc",optimal_ac),
                ("val_best_precesion",opt_pre),
                ("val_best_recall",opt_re),
                ("val_best_fscore",opt_fs),
                ("val_best_auc",opt_auc),
                ("val_best_epoch",opt_epoch),
            ])

            rowd = OrderedDict([ (str(k)+'-fold/'+_k,_v) for _k, _v in rowd.items()])
            wandb.log(rowd)
        
        # save checkpoint
        random_state = {
            'np': np.random.get_state(),
            'torch': torch.random.get_rng_state(),
            'py': random.getstate(),
            'loader': train_loader.sampler.generator.get_state() if args.fix_loader_random else '',
        }
        ckp = {
            'model': model.state_dict(),
            'lr_sche': scheduler.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch+1,
            'k': k,
            'early_stop': early_stopping.state_dict(),
            'random': random_state,
            'ckc_metric': [acs,pre,rec,fs,auc,te_auc,te_fs],
            'val_best_metric': [optimal_ac, opt_pre, opt_re, opt_fs, opt_auc,opt_epoch],
            'wandb_id': wandb.run.id if args.wandb else '',
        }
        if not args.no_log:
            torch.save(ckp, os.path.join(args.model_path, 'ckp.pt'))

        if stop:
            break
    
    # test
    if not args.no_log:
        best_std = torch.load(os.path.join(args.model_path, 'fold_{fold}_model_best_auc.pt'.format(fold=k)))
        info = model.load_state_dict(best_std['model'])
        print(info)

    accuracy, auc_value,auc_value_macro,auc_value_weighted, precision,precision_macro,precision_weighted, recall,recall_macro,recall_weighted, fscore,fscore_macro,fscore_weighted,three_scores,test_loss_log, bag_labels, bag_predictions, file_name,y_pred = test_final(args,model,test_loader,device,criterion)

    bag_predictions_array = np.array(bag_predictions)
    bag_predictions_list = bag_predictions_array.tolist()
    # TESTsave = {
    #     "Name": file_name,
    #     "True Label": bag_labels,
    #     "Prediction": y_pred,
    #     "Probability Class 0": bag_predictions_list[:, 0],
    #     "Probability Class 1": bag_predictions_list[:, 1],
    #     "Probability Class 2": bag_predictions_list[:, 2]
    # }

    TESTsave = {
        "Name": file_name,
        "True Label": bag_labels,
        "Prediction": y_pred,
    }
    for i in range(len(bag_predictions_list[0])):
        TESTsave[f'Probability Class {i}'] = [prob[i] for prob in bag_predictions_list]
    df_test = pd.DataFrame(TESTsave)
    testResultsPath = os.path.join(args.model_path, 'TEST_RESULT_PATIENT_BASED_FOLD_' + str(k) + '.csv')
    df_test.to_csv(testResultsPath, index=False)
    if args.wandb:
        wandb.log({
            "test_acc":accuracy,
            "test_precesion":precision,
            "test_precision_macro":precision,
            "test_precision_weighted":precision,
            "test_recall_micro":recall,
            "test_recall_macro":recall_macro,
            "test_recall_weighted":recall_weighted,
            "test_fscore_micro":fscore,
            "test_fscore_macro":fscore_macro,
            "test_fscore_weighted":fscore_weighted,
            "test_auc_micro":auc_value,
            "auc_value_macro":auc_value_macro,
            "auc_value_weighted":auc_value_weighted,
            "test_loss":test_loss_log,
            "three_scores":three_scores
        })
    if not args.no_log:
        print('\n Optimal accuracy: %.3f ,auc: %.3f,Optimal precision: %.3f,Optimal recall: %.3f,Optimal fscore: %.3f' % (optimal_ac,opt_auc,opt_pre,opt_re,opt_fs))
        # acs, pre, pre_mac, pre_wei, rec, rec_mac, rec_wei, fs, fs_mac, fs_wei, auc, auc_mac, auc_wei, te_auc, te_auc_mac, te_auc_wei, te_fs, te_fs_mac, te_fs_wei
        # accuracy, auc_value, auc_value_macro, auc_value_weighted, precision, precision_macro, precision_weighted, recall, recall_macro, recall_weighted, fscore, fscore_macro, fscore_weighted, three_scores
    acs.append(accuracy)
    pre.append(precision)
    pre_mac.append(precision_macro)
    pre_wei.append(precision_weighted)
    rec.append(recall)
    rec_mac.append(recall_macro)
    rec_wei.append(recall_weighted)
    fs.append(fscore)
    fs_mac.append(fscore_macro)
    fs_wei.append(fscore_weighted)
    auc.append(auc_value)
    auc_mac.append(auc_value_macro)
    auc_wei.append(auc_value_weighted)
    bag_l.append(bag_labels)
    bag_pre.append(bag_predictions_list)
    p_name.append(file_name)
    pred_int.append(y_pred)

    if args.always_test:
        te_auc.append(opt_te_auc)
        te_fs.append(opt_te_fs)
        
    return [acs, pre, pre_mac, pre_wei, rec, rec_mac, rec_wei, fs, fs_mac, fs_wei, auc, auc_mac, auc_wei,te_auc,te_fs,bag_l,bag_pre,p_name, pred_int]

def train_loop(args,model,loader,optimizer,device,amp_autocast,criterion,loss_scaler,scheduler,k,mm_sche,epoch):
    start = time.time()
    loss_cls_meter = AverageMeter()
    loss_cl_meter = AverageMeter()
    patch_num_meter = AverageMeter()
    keep_num_meter = AverageMeter()
    mm_meter = AverageMeter()
    train_loss_log = 0.
    model.train()

    for i, data in enumerate(loader):
        optimizer.zero_grad()
        if isinstance(data[0],(list,tuple)):
            for i in range(len(data[0])):
                data[0][i] = data[0][i].to(device)
            bag=data[0]
            batch_size=data[0][0].size(0)

        else:
            bag=data[0].to(device)
            batch_size=bag.size(0)

        label=data[1].to(device)


        with amp_autocast():
            if args.patch_shuffle:
                bag = patch_shuffle(bag,args.shuffle_group)
            elif args.group_shuffle:
                bag = group_shuffle(bag,args.shuffle_group)

            if args.model == 'htsfomer':
                train_logits,patch_num,keep_num = model(bag,None,None,i=epoch*len(loader)+i)
                cls_loss = 0.
            elif args.model == 'dsmil':
                _, train_logits = model(bag)
                cls_loss, patch_num, keep_num = 0., 0., 0.
            elif args.model == 'abmil':
                train_logits, Y_prob, Y_hat, A = model(bag)
                cls_loss, patch_num, keep_num = 0., 0., 0.
            elif args.model in ('clam_sb','clam_mb'):
                train_logits,cls_loss,patch_num = model(bag,label,criterion)
                keep_num = patch_num
            else:
                train_logits = model(bag)
                cls_loss,patch_num,keep_num = 0.,0.,0.

            if args.loss == 'ce':
                logit_loss = criterion(train_logits.view(batch_size,-1),label)
            elif args.loss == 'bce':
                logit_loss = criterion(train_logits.view(batch_size,-1),one_hot(label.view(batch_size,-1).float(),num_classes=3))

        train_loss = args.cls_alpha * logit_loss +  cls_loss*args.cl_alpha

        train_loss = train_loss / args.accumulation_steps
        if args.clip_grad > 0.:
            dispatch_clip_grad(
                model_parameters(model),
                value=args.clip_grad, mode='norm')

        if (i+1) % args.accumulation_steps == 0:
            train_loss.backward()
            optimizer.step()
            if args.lr_supi and scheduler is not None:
                scheduler.step()
            if args.model == 'htsfomer':
                if mm_sche is not None:
                    mm = mm_sche[epoch*len(loader)+i]
                else:
                    mm = args.mm
            else:
                mm = 0.

        loss_cls_meter.update(logit_loss,1)
        loss_cl_meter.update(cls_loss,1)
        patch_num_meter.update(patch_num,1)
        keep_num_meter.update(keep_num,1)
        mm_meter.update(mm,1)

        if i % args.log_iter == 0 or i == len(loader)-1:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)
            rowd = OrderedDict([
                ('cls_loss',loss_cls_meter.avg),
                ('lr',lr),
                ('cl_loss',loss_cl_meter.avg),
                ('patch_num',patch_num_meter.avg),
                ('keep_num',keep_num_meter.avg),
                ('mm',mm_meter.avg),
            ])
            if not args.no_log:
                print('[{}/{}] logit_loss:{}, cls_loss:{},  patch_num:{}, keep_num:{} '.format(i,len(loader)-1,loss_cls_meter.avg,loss_cl_meter.avg,patch_num_meter.avg, keep_num_meter.avg))
            rowd = OrderedDict([ (str(k)+'-fold/'+_k,_v) for _k, _v in rowd.items()])
            if args.wandb:
                wandb.log(rowd)

        train_loss_log = train_loss_log + train_loss.item()

    end = time.time()
    train_loss_log = train_loss_log/len(loader)
    if not args.lr_supi and scheduler is not None:
        scheduler.step()
    
    return train_loss_log,start,end

def val_loop(args,model,loader,device,criterion,early_stopping,epoch):

    model.eval()
    loss_cls_meter = AverageMeter()
    bag_logit, bag_labels=[], []

    with torch.no_grad():
        for i, data in enumerate(loader):
            if len(data[1]) > 1:
                bag_labels.extend(data[1].tolist())
            else:
                bag_labels.append(data[1].item())

            if isinstance(data[0],(list,tuple)):
                for i in range(len(data[0])):
                    data[0][i] = data[0][i].to(device)
                bag=data[0]
                batch_size=data[0][0].size(0)
            else:
                bag=data[0].to(device)  # b*n*1024
                batch_size=bag.size(0)

            label=data[1].to(device)
            if args.model in ('htsfomer'):
                test_logits = model.forward_test(bag)
            elif args.model == 'dsmil':
                _, test_logits = model(bag)
            elif args.model == 'abmil':
                test_logits, Y_prob, Y_hat, A = model(bag)
            elif args.model in ('clam_sb','clam_mb'):
                test_logits, _, _ = model(bag,label)
            else:
                test_logits = model(bag)
            # print("---------------test_logits-------------------")
            # print(test_logits)
            # print("-----------softmax----test_logits-------------------")
            # print(torch.softmax(test_logits,dim=-1))
            # print("-----------softmax----test_logits---numpy----------------")
            # print(torch.softmax(test_logits,dim=-1).cpu().squeeze().numpy())
            if args.loss == 'ce':
                if (args.model == 'dsmil' and args.ds_average) or (args.model == 'htsfomer' and isinstance(test_logits,(list,tuple))):
                    test_loss = criterion(test_logits.view(batch_size,-1),label)
                    bag_logit.append((1/3*torch.softmax(test_logits[1],dim=-1)+1/3*torch.softmax(test_logits[0],dim=-1)+1/3*torch.softmax(test_logits[2],dim=-1)).cpu().squeeze().numpy())
                else:
                    test_loss = criterion(test_logits.view(batch_size,-1),label)
                    if batch_size > 1:
                        bag_logit.extend(torch.softmax(test_logits,dim=-1).cpu().squeeze().numpy())
                    else:
                        bag_logit.append(torch.softmax(test_logits,dim=-1).cpu().squeeze().numpy())
            elif args.loss == 'bce':
                if args.model == 'dsmil' and args.ds_average:
                    test_loss = criterion(test_logits.view(batch_size,-1),label)
                    bag_logit.append((1/3*torch.softmax(test_logits[1], dim=-1)+1/3*torch.softmax(test_logits[0], dim=-1)+1/3*torch.softmax(test_logits[2],dim=-1)).cpu().squeeze().numpy())
                    # bag_logit.append((0.5*torch.sigmoid(test_logits[1])+0.5*torch.sigmoid(test_logits[0]).cpu().squeeze().numpy()))
                else:
                    test_loss = criterion(test_logits.view(batch_size,-1),label.view(batch_size,-1).float())
                    
                    bag_logit.append(torch.sigmoid(test_logits).cpu().squeeze().numpy())

            loss_cls_meter.update(test_loss,1)
    
    # save the log file
    # print("-----------bag_labels-----------------")
    # print(bag_labels)
    # print(len(bag_labels))
    #
    # print("-----------bag_logit-----------------")
    # print(bag_logit)
    # print(len(bag_logit))
    # accuracy, auc_value, precision, recall, fscore = five_scores(bag_labels, bag_logit)
    accuracy, auc_value, auc_value_macro, auc_value_weighted, precision, precision_macro, precision_weighted, recall, recall_macro, recall_weighted, fscore, fscore_macro, fscore_weighted, three_scores = five_scores(bag_labels, bag_logit)
    
    # early stop
    if early_stopping is not None:
        early_stopping(epoch,-auc_value,model)
        stop = early_stopping.early_stop
    else:
        stop = False
    # return stop,accuracy, auc_value, precision, recall, fscore, loss_cls_meter.avg
    return stop,accuracy, auc_value, auc_value_macro, auc_value_weighted, precision, precision_macro, precision_weighted, recall, recall_macro, recall_weighted, fscore, fscore_macro, fscore_weighted, three_scores, loss_cls_meter.avg

def test(args,model,loader,device,criterion):
    model.eval()
    test_loss_log = 0.
    bag_logit, bag_labels = [], []

    with torch.no_grad():
        for i, data in enumerate(loader):
            if len(data[1]) > 1:
                bag_labels.extend(data[1].tolist())
            else:
                bag_labels.append(data[1].item())
                
            if isinstance(data[0],(list,tuple)):
                for i in range(len(data[0])):
                    data[0][i] = data[0][i].to(device)
                bag=data[0]
                batch_size=data[0][0].size(0)
            else:
                bag=data[0].to(device)  # b*n*1024
                batch_size=bag.size(0)

            label=data[1].to(device)
            if args.model in ('htsfomer','pure'):
                test_logits = model.forward_test(bag)
            elif args.model == 'dsmil':
                _, test_logits = model(bag)
            elif args.model == 'abmil':
                test_logits, Y_prob, Y_hat, A = model(bag)

            else:
                test_logits = model(bag)

            if args.loss == 'ce':
                if (args.model == 'dsmil' and args.ds_average) or (args.model == 'htsfomer' and isinstance(test_logits,(list,tuple))):
                    test_loss = criterion(test_logits.view(batch_size,-1),label)
                    bag_logit.append((1 / 3 * torch.softmax(test_logits[1], dim=-1) + 1 / 3 * torch.softmax(test_logits[0], dim=-1) + 1 / 3 * torch.softmax(test_logits[2],dim=-1)).cpu().squeeze().numpy())
                    # bag_logit.append((0.5*torch.softmax(test_logits[1],dim=-1)+0.5*torch.softmax(test_logits[0],dim=-1)).cpu().squeeze().numpy())
                else:
                    test_loss = criterion(test_logits.view(batch_size,-1),label)
                    if batch_size > 1:
                        bag_logit.extend(torch.softmax(test_logits,dim=-1).cpu().squeeze().numpy())
                    else:
                        bag_logit.append(torch.softmax(test_logits,dim=-1).cpu().squeeze().numpy())
            elif args.loss == 'bce':
                if args.model == 'dsmil' and args.ds_average:
                    test_loss = criterion(test_logits.view(batch_size,-1),label)
                    bag_logit.append((1 / 3 * torch.softmax(test_logits[1], dim=-1) + 1 / 3 * torch.softmax(test_logits[0], dim=-1) + 1 / 3 * torch.softmax(test_logits[2],dim=-1)).cpu().squeeze().numpy())
                    # bag_logit.append((0.5*torch.sigmoid(test_logits[1])+0.5*torch.sigmoid(test_logits[0]).cpu().squeeze().numpy()))
                else:
                    test_loss = criterion(test_logits.view(batch_size,-1),label.view(1,-1).float())
                bag_logit.append(torch.sigmoid(test_logits).cpu().squeeze().numpy())

            test_loss_log = test_loss_log + test_loss.item()
    
    # save the log file
    # accuracy, auc_value, precision, recall, fscore = five_scores(bag_labels, bag_logit)
    accuracy, auc_value,auc_value_macro,auc_value_weighted, precision,precision_macro,precision_weighted, recall,recall_macro,recall_weighted, fscore,fscore_macro,fscore_weighted,three_scores = five_scores(bag_labels, bag_logit)
    test_loss_log = test_loss_log/len(loader)

    return accuracy, auc_value,auc_value_macro,auc_value_weighted, precision,precision_macro,precision_weighted, recall,recall_macro,recall_weighted, fscore,fscore_macro,fscore_weighted,three_scores,test_loss_log


def test_final(args, model, loader, device, criterion):
    model.eval()
    test_loss_log = 0.
    bag_logit, bag_labels, file_name = [], [], []

    with torch.no_grad():
        for i, data in enumerate(loader):
            if len(data[1]) > 1:
                bag_labels.extend(data[1].tolist())
                file_name.extend(data[2].tolist())
            else:
                bag_labels.append(data[1].item())
                file_name.extend(list(data[2]))
            if isinstance(data[0], (list, tuple)):
                for i in range(len(data[0])):
                    data[0][i] = data[0][i].to(device)
                bag = data[0]
                batch_size = data[0][0].size(0)
            else:
                bag = data[0].to(device)  # b*n*1024
                batch_size = bag.size(0)

            label = data[1].to(device)
            if args.model in ('htsfomer'):
                test_logits = model.forward_test(bag)
            elif args.model == 'dsmil':
                test_logits, _ = model(bag)
            elif args.model in ('clam_sb','clam_mb'):
                test_logits, _, _ = model(bag,label)
            elif args.model == 'abmil':
                test_logits, Y_prob, Y_hat, A = model(bag)
            else:
                test_logits = model(bag)

            if args.loss == 'ce':
                if (args.model == 'dsmil' and args.ds_average) or (args.model == 'htsfomer' and isinstance(test_logits, (list, tuple))):
                    test_loss = criterion(test_logits.view(batch_size, -1), label)
                    bag_logit.append((1 / 3 * torch.softmax(test_logits[1], dim=-1) + 1 / 3 * torch.softmax(test_logits[0], dim=-1) + 1 / 3 * torch.softmax(test_logits[2],dim=-1)).cpu().squeeze().numpy())
                    # bag_logit.append((0.5*torch.softmax(test_logits[1],dim=-1)+0.5*torch.softmax(test_logits[0],dim=-1)).cpu().squeeze().numpy())
                else:
                    test_loss = criterion(test_logits.view(batch_size, -1), label)
                    if batch_size > 1:
                        bag_logit.extend(torch.softmax(test_logits, dim=-1).cpu().squeeze().numpy())
                    else:
                        bag_logit.append(torch.softmax(test_logits, dim=-1).cpu().squeeze().numpy())
            elif args.loss == 'bce':
                if args.model == 'dsmil' and args.ds_average:
                    test_loss = criterion(test_logits.view(batch_size, -1), label)
                    bag_logit.append((1 / 3 * torch.softmax(test_logits[1], dim=-1) + 1 / 3 * torch.softmax(
                        test_logits[0], dim=-1) + 1 / 3 * torch.softmax(test_logits[2],
                                                                        dim=-1)).cpu().squeeze().numpy())
                    # bag_logit.append((0.5*torch.sigmoid(test_logits[1])+0.5*torch.sigmoid(test_logits[0]).cpu().squeeze().numpy()))
                else:
                    test_loss = criterion(test_logits.view(batch_size, -1), label.view(1, -1).float())
                bag_logit.append(torch.sigmoid(test_logits).cpu().squeeze().numpy())

            test_loss_log = test_loss_log + test_loss.item()

    # save the log file
    # accuracy, auc_value, precision, recall, fscore = five_scores(bag_labels, bag_logit)
    # accuracy, auc_value,auc_value_macro,auc_value_weighted, precision,precision_macro,precision_weighted, recall,recall_macro,recall_weighted, fscore,fscore_macro,fscore_weighted,three_scores = five_scores(bag_labels, bag_logit, file_name)
    accuracy, auc_value, auc_value_macro, auc_value_weighted, precision, precision_macro, precision_weighted, recall, recall_macro, recall_weighted, fscore, fscore_macro, fscore_weighted, three_scores,bag_labels, bag_predictions, y_pred = five_scores_final(bag_labels, bag_logit)
    test_loss_log = test_loss_log / len(loader)

    return accuracy, auc_value, auc_value_macro, auc_value_weighted, precision, precision_macro, precision_weighted, recall, recall_macro, recall_weighted, fscore, fscore_macro, fscore_weighted, three_scores, test_loss_log,bag_labels, bag_predictions, file_name,y_pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MIL Training Script')

    # Dataset 
    parser.add_argument('--datasets', default='TCGA', type=str, help='[cpgea, tcga]')
    parser.add_argument('--dataset_root', default='/TCGA/FEATURES', type=str, help='Dataset root path')
    parser.add_argument('--tcga_max_patch', default=-1, type=int, help='Max Number of patch in TCGA [-1]')
    parser.add_argument('--fix_loader_random', action='store_true', help='Fix random seed of dataloader')
    parser.add_argument('--fix_train_random', action='store_true', help='Fix random seed of Training')
    parser.add_argument('--val_ratio', default=0., type=float, help='Val-set ratio')
    parser.add_argument('--fold_start', default=0, type=int, help='Start validation fold [0]')
    parser.add_argument('--cv_fold', default=3, type=int, help='Number of cross validation fold [3]')
    parser.add_argument('--persistence', action='store_true', help='Load data into memory') 
    parser.add_argument('--same_psize', default=0, type=int, help='Keep the same size of all patches [0]')

    # Train
    parser.add_argument('--cls_alpha', default=1.0, type=float, help='Main loss alpha')
    parser.add_argument('--auto_resume', action='store_true', help='Resume from the auto-saved checkpoint')
    parser.add_argument('--num_epoch', default=200, type=int, help='Number of total training epochs [200]')
    parser.add_argument('--early_stopping', action='store_false', help='Early stopping')
    parser.add_argument('--max_epoch', default=200, type=int, help='Number of max training epochs in the earlystopping [130]')
    parser.add_argument('--n_classes', default=3, type=int, help='Number of classes')
    parser.add_argument('--batch_size', default=1, type=int, help='Number of batch size')
    parser.add_argument('--loss', default='ce', type=str, help='Classification Loss [ce, bce]')
    parser.add_argument('--opt', default='adam', type=str, help='Optimizer [adam, adamw]')
    parser.add_argument('--save_best_model_stage', default=0., type=float, help='See DTFD')
    parser.add_argument('--model', default='htsfomer', type=str, help='Model name')
    parser.add_argument('--seed', default=2024, type=int, help='random number [2025]' )
    parser.add_argument('--lr', default=2e-4, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--lr_sche', default='cosine', type=str, help='Deacy of learning rate [cosine, step, const]')
    parser.add_argument('--lr_supi', action='store_true', help='LR scheduler update per iter')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--accumulation_steps', default=1, type=int, help='Gradient accumulate')
    parser.add_argument('--clip_grad', default=.0, type=float, help='Gradient clip')
    parser.add_argument('--always_test', action='store_true', help='Test model in the training phase')

    # Model
    # Other models
    parser.add_argument('--ds_average', action='store_true', help='DSMIL hyperparameter')
    # Our
    parser.add_argument('--baseline', default='selfattn', type=str, help='Baselin model [attn,selfattn]')
    parser.add_argument('--act', default='relu', type=str, help='Activation func in the projection head [gelu,relu]')
    parser.add_argument('--dropout', default=0.25, type=float, help='Dropout in the projection head')
    parser.add_argument('--n_heads', default=8, type=int, help='Number of head in the MSA')
    parser.add_argument('--da_act', default='relu', type=str, help='Activation func in the DAttention [gelu,relu]')

    # Shuffle
    parser.add_argument('--patch_shuffle', action='store_true', help='2-D group shuffle')
    parser.add_argument('--group_shuffle', action='store_true', help='Group shuffle')
    parser.add_argument('--shuffle_group', default=0, type=int, help='Number of the shuffle group')

    # MMILT
    parser.add_argument('--mode', default='embed', type=str, help='Optimizer [random,coords, seq, embed,idx]')
    parser.add_argument('--in_chans', default=1024, type=int, help='in_chans')
    parser.add_argument('--embed_dim', default=512, type=int, help='embed_dim')
    parser.add_argument('--num_subbags', default=3, type=int, help='num_subbags')
    parser.add_argument('--num_msg', default=3, type=int, help='num_msg')

    # htsfomer
    parser.add_argument('--mask_ratio', default=0., type=float, help='Random mask ratio')
    parser.add_argument('--mask_ratio_l', default=0., type=float, help='Low attention mask ratio')
    parser.add_argument('--mask_ratio_h', default=0., type=float, help='High attention mask ratio')
    parser.add_argument('--mask_ratio_hr', default=1., type=float, help='Randomly high attention mask ratio')
    parser.add_argument('--mrh_sche', action='store_true', help='Decay of HAM')
    parser.add_argument('--msa_fusion', default='vote', type=str, help='[mean,vote]')
    parser.add_argument('--attn_layer', default=0, type=int)
    
    # Siamese framework
    parser.add_argument('--cl_alpha', default=0., type=float, help='Auxiliary loss alpha')
    parser.add_argument('--temp_t', default=0.1, type=float, help='Temperature')
    parser.add_argument('--mm', default=0.9999, type=float, help='Ema decay [0.9997]')
    parser.add_argument('--mm_final', default=1., type=float, help='Final ema decay [1.]')
    parser.add_argument('--mm_sche', action='store_true', help='Cosine schedule of ema decay')

    # Misc
    parser.add_argument('--title', default='transmil_101_sml80h3-0r50_mmcos_is', type=str, help='Title of exp')
    parser.add_argument('--project', default='mil_trans_c16_self_flod5weight_test', type=str, help='Project name of exp')
    parser.add_argument('--log_iter', default=100, type=int, help='Log Frequency')
    parser.add_argument('--amp', action='store_true', help='Automatic Mixed Precision Training')
    parser.add_argument('--wandb', action='store_true', help='Weight&Bias')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of workers in the dataloader')
    parser.add_argument('--no_log', action='store_true', help='Without log')
    parser.add_argument('--model_path', type=str, help='Output path')
    parser.add_argument('--num_feats', default=512, type=int, help='feats size')
    parser.add_argument('--num_feats_ab', default=1024, type=int, help='feats size')

    ## mambamil

    parser.add_argument('--mambamil_rate', type=int, default=10, help='mambamil_rate')
    parser.add_argument('--mambamil_layer', type=int, default=2, help='mambamil_layer')
    parser.add_argument('--mambamil_type', type=str, default='SRMamba', choices=['Mamba', 'BiMamba', 'SRMamba'],
                        help='mambamil_type')

    args = parser.parse_args()

    
    if not os.path.exists(os.path.join(args.model_path,args.project)):
        os.mkdir(os.path.join(args.model_path,args.project))
    args.model_path = os.path.join(args.model_path,args.project,args.title)
    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)

    if args.model == 'clam_sb':
        args.cls_alpha= .7
        args.cl_alpha = .3
    elif args.model == 'clam_mb':
        args.cls_alpha= .7
        args.cl_alpha = .3
    elif args.model == 'dsmil':
        args.cls_alpha = 0.5
        args.cl_alpha = 0.5


    if args.datasets == 'tcga':
        args.num_workers = 0
        args.always_test = True

    wandb.init(project=args.project, name=args.title, config=args, dir=os.path.join(args.model_path))

    print(args)

    localtime = time.asctime( time.localtime(time.time()) )
    print(localtime)
    main(args=args)
