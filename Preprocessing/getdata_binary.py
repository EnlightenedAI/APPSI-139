import os
import json
import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from pprint import pprint
import random
import numpy as np
dataset='all_data'
path= '../'+dataset+'/doccano/'
labelDataFile = 'TCSI_pp/preprocessing/label_config.json'
labels_text=open(labelDataFile, 'r', encoding='utf_8').read()
labels_list = json.loads(labels_text)
class_label={}

for i, label in enumerate(labels_list):
    class_label.update({label['text']:i+1})

testrate=0.10
def merge_json(path_results, path_merges):
    """
    主要功能是实现一个目录下的多个json文件合并为一个json文件。
    :param path_results:
    :param path_merges:
    :return:
    """
    sentences=[]
    ll=[]
    f_errors=open('errorsdds.json','w',encoding='utf-8')
    merges_file = os.path.join(path_merges, "bas_fund_transaction.json")
    with open(merges_file, "w", encoding="utf-8") as f0:
        for i,file in enumerate(os.listdir(path_results)):
            print(i,file)
            if i<150 and file!='.DS_Store':
                with open(os.path.join(path_results, file), "r", encoding="utf-8") as f1:
                    for line in tqdm.tqdm(f1):
                        line_dict = json.loads(line)
                        l = [0] * (len(class_label) + 1)
                        if len(line_dict['label'])>1 and ('Important' in line_dict['label'] or "Risk"in line_dict['label'] or "Sensitive" in line_dict['label']):
                            l[12]=1
                        else:
                            for lab in line_dict["label"] :
                                l[class_label[lab]]=1
                        ll.append(l)
                        sens= line_dict["text"]
                        sens=json.dumps({"text":sens}, ensure_ascii=False)
                        
                        sentences.append([sens, l])
                        js = json.dumps(line_dict, ensure_ascii=False)
                        f0.write(js + '\n')
                print(len(sentences))
                f1.close()
    f0.close()
    return sentences
if __name__ == '__main__':
    path_results, path_merges = 'TCSI_pp/preprocessing/doccano_new', 'TCSI_pp/preprocessing/results_con10'
    if not os.path.exists(path_merges):
        os.mkdir(path_merges)
    sentences = merge_json(path_results, path_merges)
    print(len(sentences))

    for i in range(13,15):
        X = np.array([], dtype=np.str_)
        y = np.array([])
        for lens in sentences:
            if i==0:
                X = np.append(X, lens[0])
                if  lens[1][12]==0:
                    y = np.append(y,0)
                else:
                    y = np.append(y, lens[1][i])
            elif i==12:
                X = np.append(X, lens[0])
                y = np.append(y, lens[1][i])
            else:
                if lens[1][12]==1:
                    X=np.append(X,lens[0])
                    y=np.append(y,lens[1][i])
        sss = StratifiedShuffleSplit(n_splits=1, test_size=testrate*2, random_state=0)
        sss.get_n_splits(X, y)
        for train_index, test_dev_index in sss.split(X, y):
            X_train, X_test_dev = X[train_index], X[test_dev_index]
            print(X_train.shape)
            print(X_test_dev.shape)
            y_train, y_test_dev = y[train_index], y[test_dev_index]
        from sklearn.utils import resample

        X_train_a = np.array([], dtype=np.str_)
        y_train_a = np.array([])
        X_train_o = np.array([], dtype=np.str_)
        y_train_o = np.array([])
        for sentence,y_a in zip(X_train,y_train):
            x_a=json.loads(sentence)
            X_train_a = np.append(X_train_a,x_a["text"])
            y_train_a = np.append(y_train_a,y_a)
            X_train_o = np.append(X_train_o, x_a["text"])
            y_train_o = np.append(y_train_o, y_a)

        sss_T = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
        sss_T.get_n_splits(X_test_dev, y_test_dev)
        for dev_index, test_index in sss_T.split(X_test_dev, y_test_dev):
            X_dev, X_test = X_test_dev[dev_index], X_test_dev[test_index]
            y_dev, y_test = y_test_dev[dev_index], y_test_dev[test_index]
            print(X_dev.shape)
            print(X_test.shape)

        X_dev_a = np.array([], dtype=np.str_)
        y_dev_a = np.array([])
        X_dev_o = np.array([], dtype=np.str_)
        y_dev_o = np.array([])
        for sentence,y_a in zip(X_dev,y_dev):
            x_a=json.loads(sentence)
            X_dev_a = np.append(X_dev_a,x_a["text"])
            y_dev_a = np.append(y_dev_a,y_a)
            X_dev_o = np.append(X_dev_o, x_a["text"])
            y_dev_o = np.append(y_dev_o, y_a)


        X_test_o = np.array([], dtype=np.str_)
        y_test_o = np.array([])
        for sentence, y_a in zip(X_test, y_test):
            x_a = json.loads(sentence)
            X_test_o = np.append(X_test_o, x_a["text"])
            y_test_o = np.append(y_test_o, y_a)


        data_url=path_merges+'/class'+str(i)



        if not os.path.exists(data_url):
            os.mkdir(data_url)

        with open(data_url+'/train.txt', "w", encoding="utf-8") as train_text:
            for j in range(X_train_o.shape[0]):
                train_text.write(str(X_train_o[j]) + '\t' + str(int(y_train_o[j])) + "\n")
        train_text.close()

        with open(data_url+ '/dev.txt', "w", encoding="utf-8") as dev_text:
            for j in range(X_dev_o.shape[0]):
                dev_text.write(str(X_dev_o[j]) + '\t' + str(int(y_dev_o[j])) + "\n")
        dev_text.close()

        with open(data_url+ '/test.txt', "w", encoding="utf-8") as test_text:
            for j in range(X_test_o.shape[0]):
                test_text.write(str(X_test_o[j]) + '\t' + str(int(y_test_o[j])) + "\n")
        test_text.close()


