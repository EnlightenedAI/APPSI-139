# coding: UTF-8
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
from pytorch_pretrained.optimization import BertAdam
from sklearn.metrics import recall_score,f1_score,precision_score,roc_auc_score
# from loss.focalloss import BCEFocalLoss
from loss.focallooss import Focal_Loss
import os
from sklearn.metrics import precision_recall_curve
from numpy.core.fromnumeric import argmax
# import torch.nn as nn

# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, model, train_iter, dev_iter, test_iter):
    # classID=12
    start_time = time.time()
    # model.load_state_dict(torch.load(config.load_path_1))
    # model.eval()

    # checkpoint = torch.load(config.save_path)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch = checkpoint['epoch']
    # loss = checkpoint['loss']
    # model.train()
    # for name, param in model.named_parameters():
    #     if 'bert' in name:
    #         param.requires_grad = False
    #     elif 'fc' in name:
    #         param.requires_grad = True

            
    # for name, param in model.named_parameters():
    #     print(f"Name: {name}")
    #     print(f"Shape: {param.shape}")
    #     print(f"Requires Grad: {param.requires_grad}")
    #     print()  # 打印一个空行以便区分不同的参数

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=config.learning_rate,
                         warmup=0.05,
                         t_total=len(train_iter) * config.num_epochs,
                         weight_decay=0.01)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    total_batch: int = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    dev_best_acc=0
    dev_best_f1=0
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    model.train()
    criterion_weighted=Focal_Loss(alpha=config.focalloss_rate, gamma=2) 
    for epoch in range(config.num_epochs):
        current_lr = optimizer.get_lr()
        print("当前学习率: ", current_lr[0])
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        
        for i, (trains_1, labels,lens,masks) in enumerate(train_iter):
            # print("train",trains_1)
            trains=(trains_1.to(config.device),lens.to(config.device),masks.to(config.device))
            # print(trains)
            outputs = model(trains)
            model.zero_grad()
            # print(outputs)
            # print(labels)
            # labels.format("%06d", 25)
            # loss = F.cross_entropy(outputs, labels)
            # loss =loss1(outputs, labels)
            # ----------
            # print(labels.shape)
            # labels=labels[:,config.num_classes-2:config.num_classes-1]####注意
            labels=labels.to(config.device)

            # print(labels.shape)
            # wight =labels#1
            # c = sum(wight)/wight.shape[0]#1
            # print("c",c)
            # wight[wight == 1] = c.long()#1
            # wight[wight == 0] = (1 - c).long()#1            # print(wight)
            # print(outputs)
            # print(labels.shape)
            # t = ((labels.shape[0] - labels.sum(0)) / labels.shape[0])
            # weight = torch.zeros_like(labels)
            # print(weight)
            # print(t[0])
            # weight = torch.fill_(weight,  torch.tensor(1-t[0]))
            # print(weight)
            # weight[labels > 0] = torch.tensor(t[0])
            # print(weight)
            # criterion_weighted = nn.BCELoss(weight=weight, size_average=True)
            # criterion_weighted = nn.BCELoss(weight=wight)#1
            # criterion_weighted =BCEFocalLoss()

            # criterion_weighted = nn.BCEWithLogitsLoss(weight=torch.tensor([]))
            # criterion_weighted =nn.MultiLabelSoftMarginLoss(reduction='mean')
            # criterion_weighted=nn.BCELoss(weight=torch.tensor([t,1-t]))
            # criterion_weighted = nn.BCEWithLogitsLoss(pos_weight=(y==0.).sum()/y.sum())
            # print(outputs)

                     
            loss = criterion_weighted(outputs, labels)
            # print(loss)
            # loss = criterion_weighted(outputs.to(torch.float32), labels.to(torch.float32))
            # loss1=nn.MultiLabelSoftMarginLoss(weight=None)
            # ----------
            loss.backward()
            optimizer.step()
            if total_batch % config.valstep == 0:
            # if epoch % 10 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                # predic = torch.max(outputs.data, 1)[1].cpu()#原始
                # -------
                # print("outputs.data",outputs.data)
                predic = outputs.data.argmax(axis=1)
                predict_true=outputs.data[:,1]
                # print("predic",predic)
                # print("labels", labels)
                # print("predic.shape",predic.shape)
                # print("true.shape", true.shape)
                # -------
                # precision, recall, thresholds = precision_recall_curve(np.array(true), np.array(predic))
                # #
                # # print(precision)
                # # print(recall)
                # # print(thresholds)
                # target = precision + recall
                # index = argmax(target)
                # print("p:",precision[index], "\nr:",recall[index],"\nt:",thresholds[index])
                # level=0.3
                # predic[predic >= level] = 1
                # predic[predic <level] = 0

                # print(predic)
                train_acc = metrics.accuracy_score(true.clone().detach().cpu(), predic.clone().detach().cpu())
                train_pre = precision_score(np.array(true.clone().detach().cpu()), np.array(predic.clone().detach().cpu()),average=None,zero_division=1)
                train_rec = recall_score(np.array(true.clone().detach().cpu()), np.array(predic.clone().detach().cpu()),average=None)
                train_f1 = f1_score(np.array(true.clone().detach().cpu()), np.array(predic.clone().detach().cpu()),average=None)
                # print("train_acc",train_acc)
                # train_auc=roc_auc_score(true.clone().detach().cpu(), predict_true.clone().detach().cpu())
                train_auc ="none"
                improve=''
                dev_acc, dev_loss,dev_auc,dev_pre,dev_rec,dev_f1= evaluate(config, model, dev_iter,criterion_weighted)
                # if dev_loss < dev_best_loss or dev_acc > dev_best_acc or dev_f1[1] > dev_best_f1:
                if dev_loss < dev_best_loss:
                    # if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    improve += '*'
                    im='*'
                    torch.save(model.state_dict(), config.save_path+im)
                    last_improve = total_batch
                if dev_acc > dev_best_acc:
                    dev_best_acc = dev_acc
                    improve += '+'
                    im='+'
                    torch.save(model.state_dict(), config.save_path+im)
                    last_improve = total_batch
                if dev_f1[1] > dev_best_f1:
                    dev_best_f1 = dev_f1[1]
                    improve += '&'
                    im='&'
                    torch.save(model.state_dict(), config.save_path+im)
                    last_improve = total_batch
                    # print(config.save_path)
                    # torch.save(model.state_dict(), config.save_path+)
                    # last_improve = total_batch

                # else:
                #     improve = ''
                
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.6},  Train Acc: {2:>6.4%},  Val Loss: {3:>5.6},  Val Acc: {4:>6.4%},  Time: {5} {6}'
                tra_msg='Tra_Pre: {0:>6.2%},    Tra_Rec: {1:>6.2%},    Tra_F1: {2:>6.2%},   Val_Pre: {3:>6.2%},    Val_Rec: {4:>6.2%},   Val_F1: {5:>6.2%}'
                with open(config.save_path_acc_loss,"a",encoding="utf-8") as write_loss:
                    write_loss.write(json.dumps({'Iter': str(total_batch),'Train Loss': str(loss.item()),'Train Acc': str(train_acc),
                                                    'Val Loss': str(dev_loss.item()),'Val Acc': str(dev_acc),'Time': str(time_dif),
                                                    'train_Precision': str(train_pre),'train_Recall': str(train_rec),'train_F1': str(train_f1),
                                                    'Val_Precision': str(dev_pre),'Val_Recall': str(dev_rec), 'Val_F1': str(dev_f1)},ensure_ascii=False)+ '\n')

                write_loss.close()
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                print(tra_msg.format(train_pre[1],train_rec[1],train_f1[1],dev_pre[1], dev_rec[1], dev_f1[1]))


                # with open("./CLASS_Data/output/trainput/"+config.dataset+"_"+config.model_name+"_focalloss_rate_"+str(config.focalloss_rate).replace(str(config.focalloss_rate)[1], "", 1)+".json", "a") as train_out:
                #     train_out.write(json.dumps({'train_acc':train_acc,'train_acc':train_auc,'dev_acc':dev_acc,'dev_auc':dev_auc})+'\n')
                model.train()
            total_batch += 1
            # 
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
            # scheduler.step()
    # test(config, model, test_iter, criterion_weighted)


def test(config, model,test_iter, criterion_weighted=''):
    # test

    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion ,pre,rec,f1,auc= evaluate(config, model, test_iter,criterion_weighted, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'

    # with open("./"+config.dataset+"/output/testput/"+config.dataset+"_"+config.is_model_name+"_focalloss_rate_"+str(config.focalloss_rate).replace(str(config.focalloss_rate)[1], "", 1)+".json", "a") as test_out:
    #     test_out.write(json.dumps({"test_acc":test_acc,'test_precision':pre,'test_recall':rec,'test_f1':f1,'test_auc':auc,'test_report':test_report})+'\n')
    # test_out.close()
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, criterion_weighted='', test=False):
    # classID=12
    model.eval()
    loss_total = 0
    predict_all = np.array([[]], dtype=int)
    labels_all = np.array([[]], dtype=int)
    predict_true_all = np.array([[]], dtype=int)
    # predict_all=np.empty(shape=[1, len(config.class_list)], dtype=int)
    # labels_all=np.empty(shape=[1, len(config.class_list)], dtype=int)
    # predict_all=[]
    # labels_all=[]
    with torch.no_grad(): 
        for texts, labels , lens,masks in data_iter:
            texts=(texts.to(config.device),lens.to(config.device),masks.to(config.device))
            outputs = model(texts)
            # print(outputs)
            # loss = F.cross_entropy(outputs, labels)
            #--------
            # labels = labels[:, config.num_classes - 2:config.num_classes-1 ]
            # print(labels)
            # labels = labels[:, classID]
            # t = ((labels.shape[0] - labels.sum(0)) / labels.shape[0])
            # criterion_weighted = nn.BCEWithLogitsLoss(weight=t)

            # criterion_weighted = nn.BCELoss()
            # loss = criterion_weighted(outputs.to(torch.float), labels.to(torch.float))
            # t = ((labels.shape[0] - labels.sum(0)) / labels.shape[0])
            # weight = torch.zeros_like(labels)
            # weight = torch.fill_(weight, 0.55)
            # weight[labels > 0] = 0.45
            # criterion_weighted = nn.BCELoss(weight=weight, size_average=True)
            # criterion_weighted = nn.BCEWithLogitsLoss(weight=t)
            # criterion_weighted =nn.MultiLabelSoftMarginLoss(reduction='mean')
            # criterion_weighted=nn.BCELoss(weight=torch.tensor([t,1-t]))
            # criterion_weighted = nn.BCEWithLogitsLoss(pos_weight=(y==0.).sum()/y.sum())
            labels = labels.to(config.device)
            # criterion_weighted = BCEFocalLoss()
            if criterion_weighted=='':
                criterion_weighted = Focal_Loss(alpha=config.focalloss_rate, gamma=1)
            # loss = criterion_weighted(outputs.to(torch.float32), labels.to(torch.float32))
            loss = criterion_weighted(outputs, labels)
            # loss = F.cross_entropy(outputs.to(torch.float), labels.to(torch.float))
            #--------
            loss_total += loss
            # print(labels)
            # print(labels.data)

            # labels = labels.data.cpu().numpy()
            # predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            #--------
            predic = outputs.data.argmax(axis=1)
            predict_true=outputs.data[:,1]
            # level=0.3
            # predic[predic >=level] = 1
            # predic[predic <level] = 0
            # print(predic)
            #--------
            labels_all = np.append(labels_all, labels.cpu().numpy())
            predict_all = np.append(predict_all, predic.cpu().numpy())
            predict_true_all=np.append(predict_true_all, predict_true.cpu().numpy())
            # print("label",labels_all)
            # for item in labels:
            #     labels_all.append(item.tolis  t())
            # for item in predic:
            #     predict_all.append(item.tolist())
            # labels_all=np.vstack(labels_all,labels)
            # predict_all =np.vstack(labels_all,labels)
            # labels_all=np.append(labels_all,[np.array(labels)],axis=0)
            # predict_all=np.append(predict_all,[np.array(predic)],axis=0)

    # acc=sum(sum(np.array(labels_all )== np.array(predict_all))) / np.size(predict_all)  # 所有元素
    #
    # print(sum(target == pred, 0) / np.size(target, 0))  # 每一列
    # a = 0
    #
    # for i  in range(np.size(np.array(labels_all), 0)):
    #     if (np.array(labels_all)[i, :] == np.array(predict_all)[i, :]).all():
    #         a += 1
    # # acc=a / np.size(np.array(labels_all), 0)
    # print("准确率",precision_score(np.array(labels_all),np.array(predict_all)))#,labels=None,pos_label=1,average="micro"
    # print("召回率",recall_score(np.array(labels_all),np.array(predict_all)))#,labels=None,pos_label=1,average="micro"
    # print("F1-score", f1_score(np.array(labels_all), np.array(predict_all)))
    acc = metrics.accuracy_score(np.array(labels_all.data), np.array(predict_all))
    pre = precision_score(np.array(labels_all), np.array(predict_all),average=None)
    rec = recall_score(np.array(labels_all), np.array(predict_all),average=None)
    f1 = f1_score(np.array(labels_all), np.array(predict_all),average=None)
    # print("acc",acc)
    # auc = roc_auc_score(np.array(labels_all.data), np.array(predict_true_all))
    auc='记得修改'
    # print('auc',auc)
    if test:
        # print(predict_all)
        # report = metrics.classification_report(np.array(labels_all),np.array(predict_all) , target_names=config.class_list)
        pre=precision_score(np.array(labels_all), np.array(predict_all))
        rec=recall_score(np.array(labels_all), np.array(predict_all))
        f1=f1_score(np.array(labels_all), np.array(predict_all))
        report = metrics.classification_report(np.array(labels_all.data),np.array(predict_all), digits=6)
        # print(report)
        confusion = metrics.multilabel_confusion_matrix(np.array(labels_all.data), np.array(predict_all))
        # print(confusion.shape)
        c=labels_all==predict_all
        with open('error/predict_label.txt', 'w') as fs:
            for s in predict_all:
                fs.write(str(s))
                fs.write('\n')
        fs.close()
        x = []
        with open('error/摘要.txt', 'w') as fs:
            for s in c:
                fs.write(str(s))
                fs.write('\n')
        fs.close()
        x = []
        with open('error/摘要错误句.txt', 'w', encoding='UTF-8') as fs2:
            with open('TCSI_pp/preprocessing/results/class12/train.txt', 'r', encoding='UTF-8') as fs1:
                for lin in fs1:
                    x.append(lin)
            for i, t in enumerate(c):
                # print(i, t)
                if t == False:
                    fs2.write(x[i])
        fs2.close()
        fs1.close()
        return acc, loss_total / len(data_iter), report, confusion,pre,rec,f1,auc
    return acc, loss_total / len(data_iter),auc,pre,rec,f1
