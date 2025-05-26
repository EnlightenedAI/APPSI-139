import os
import json
import tqdm
from sklearn.model_selection import StratifiedShuffleSplit,train_test_split
from skmultilearn.model_selection.iterative_stratification import iterative_train_test_split
from pprint import pprint
import random
import pdb
import numpy as np
dataset='test_data'
labelDataFile = 'TCSI_pp/preprocessing/label_config_a.json'
labels_text=open(labelDataFile, 'r', encoding='utf_8').read()
labels_list = json.loads(labels_text)

class_label={}
labelslists=[]
for i, label in enumerate(labels_list):
    class_label.update({label['text']:i+1})
    labelslists.append(label['text'])
print(class_label)
print(labelslists)
labels_list_no_sum=list(set(labelslists).intersection(set(["Important"])))
print (labels_list_no_sum)
def merge_json(path_results, path_merges):
    """
    主要功能是实现一个目录下的多个json文件合并为一个json文件。
    :param path_results:
    :param path_merges:
    :return:
    """
    sentences=[]
    ll=[]
    ll_1_11=[]
    f_errors=open('errorsdds.json','w',encoding='utf-8')
    merges_file = os.path.join(path_merges, "bas_fund_transaction.json")
    with open(merges_file, "w", encoding="utf-8") as f0:
        print(len(os.listdir(path_results)))
        for i,file in enumerate(os.listdir(path_results)):
            if i <150 and file !='.DS_Store':
                print(i,file)

                with open(os.path.join(path_results, file), "r", encoding="utf-8") as f1:
                    for line in tqdm.tqdm(f1):
                        line_dict = json.loads(line)
                        l = [0] * (len(class_label) + 1)
                        if not 'Important' in line_dict["label"] and len(line_dict["label"])>0:
                            f_errors.write(json.dumps(json.dumps({'id':line_dict["id"],"text":line_dict['text'],'label':line_dict["label"]},ensure_ascii=False)+'\n'))
                            l[0] = 1
                        elif line_dict["label"]==[]:
                            l[0] = 1
                        else:
                            for lab in line_dict["label"]:
                                if lab !='Permission Acquisition' and lab !="Cease Operation":
                                    l[class_label[lab]]=1
                            ll.append(l)
                            sens= ",".join(line_dict["text"].split())
                            sentences.append([sens, l[1:10]])
                        js = json.dumps(line_dict, ensure_ascii=False)
                        f0.write(js + '\n')
            f1.close()
        f0.close()
    return sentences
if __name__ == '__main__':
    path_results, path_merges = 'TCSI_pp/preprocessing/doccano_new', 'TCSI_pp/preprocessing/results_new10'
    if not os.path.exists(path_merges):
        os.mkdir(path_merges)
    sentences = merge_json(path_results, path_merges)
    data_url=f'{path_merges}/class_multi_a'
    if not os.path.exists(data_url):
        os.mkdir(data_url)
    X = np.array([], dtype=np.str_)
    y = np.array([])


    for i,lens in enumerate(sentences):
        if y.size==0:
            if sum(lens[1])!= 0 :
                X=lens[0]
                y=np.array(lens[1])
        else:
            if sum(lens[1]) !=0:
                X=np.append(X,lens[0])
                y=np.vstack((y,np.array(lens[1])))
    print(y.shape)
    print(X.shape)
    print(np.sum(y,0))
    print("-"*10,'开始切分数据','_'*10)
    print(len(X))
    np.random.seed(42)
    X_train,y_train, X_test_dev,  y_test_dev =iterative_train_test_split(X, y, test_size=0.2)
    print("-" * 10, '一破！卧龙出山', '_' * 10)
    np.random.seed(42)
    X_test, y_test, X_dev, y_dev =iterative_train_test_split(X_test_dev, y_test_dev, test_size=0.5)
    print(X_dev.shape)
    print(X_test.shape)
    

    with open(data_url+'/train.txt', "w", encoding="utf-8") as train_text:
        for j in range(X_train.shape[0]):
            train_text.write(str(X_train[j]) + '\t' + str(y_train[j].tolist()) + "\n")
    train_text.close()

    with open(data_url+ '/dev.txt', "w", encoding="utf-8") as dev_text:
        for j in range(X_dev.shape[0]):
            dev_text.write(str(X_dev[j]) + '\t' + str(y_dev[j].tolist()) + "\n")
    dev_text.close()

    with open(data_url+ '/test.txt', "w", encoding="utf-8") as test_text:
        for j in range(X_test.shape[0]):
            test_text.write(str(X_test[j]) + '\t' + str(y_test[j].tolist()) + "\n")
    test_text.close()


