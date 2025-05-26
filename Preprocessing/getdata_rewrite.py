import os
import json
import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from pprint import pprint
import random
import numpy as np
import pdb
class_label={}

testrate=0.10
def merge_json(path_results, path_merges):
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
                        if len(line_dict['label'])>0 :
                            sentence={'id':line_dict['id'],'sentence':line_dict['text'],'rewrite':line_dict['label'][0]}
                            sentences.append(sentence)
                f1.close()
    f0.close()
    return sentences
import random
if __name__ == '__main__':
    path_results, path_merges = 'TCSI_pp/preprocessing/rewrite', 'TCSI_pp/preprocessing/results_rewrite'
    if not os.path.exists(path_merges):
        os.mkdir(path_merges)
    sentences = merge_json(path_results, path_merges)
    print(len(sentences))
    print(sentences[0])
    random.shuffle(sentences)
    val_test_size = int(len(sentences) * 0.2)
    val_test = random.sample(sentences, val_test_size)
    trains = [x for x in sentences if x not in val_test]

    val = random.sample(val_test, int(val_test_size*0.5))

    test=[x for x in val_test if x not in val]
    with open('TCSI_pp/preprocessing/results_rewrite' +'/train.json', 'w', encoding='utf-8') as f:
        for sen in trains:
            f.write(json.dumps(sen,ensure_ascii=False) + '\n')
    with open('TCSI_pp/preprocessing/results_rewrite' +'/dev.json', 'w', encoding='utf-8') as f:
        for sen in val:
            f.write(json.dumps(sen,ensure_ascii=False) + '\n')
    with open('TCSI_pp/preprocessing/results_rewrite' +'/test.json', 'w', encoding='utf-8') as f:
        for sen in test:
            f.write(json.dumps(sen,ensure_ascii=False) + '\n')        


