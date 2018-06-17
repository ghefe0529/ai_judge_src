# -*- coding: utf-8 -*-
# @Date    : 2018-06-12 10:35:32
# @Author  : GEFE (gh_efe@163.com)
# @Version : 1.0.0
# @Describe : 处理特征

import pandas as pd 
import jieba

all_train_path = r'../../data/corpus/input/train.txt'
all_stop_path = r'../../data/corpus/input/stop.txt'
pre_train_path = r'../../data/corpus/output/train.csv'
pre_id_context_path = r'../../data/corpus/output/id_context.csv'
pre_id_penalty_path = r'../../data/corpus/output/id_penalty.csv'
pre_id_violatelow_path = r'../../data/corpus/output/id_violatelow.csv'
pre_context_path = r'../../data/corpus/output/context.csv'

def read_stop_words():
    stop = []
    with open(all_stop_path, encoding='utf-8') as f:
        for line in f:
            stop.extend(line.split())
    return stop

def jieba_cut_text(text, stop):
    all_words = jieba.cut(text)
    useful_words = []
    for ele in all_words:
        # print(ele.split())
        if ele not in stop:
            useful_words.extend(ele.split())
    # print(useful_words)
    return ' '.join(useful_words)

def read_all_train():
    output_data = []
    stop = read_stop_words()
    count = 0
    with open(all_train_path, encoding='utf-8') as f:
        for line in f:
            line = line.replace('\n','')
            data = line.split('\t')
            tmp = dict()
            tmp['id']=data[0]
            tmp['context'] = jieba_cut_text(data[1], stop)
            tmp['penalty'] = data[2]
            tmp['violatelow'] = data[3].split(',')
            output_data.append(tmp)
            if count%1000 == 0:
                print("百分之" , ((count/120000)*100))
            count += 1
    # print(output_data)
    output_data = pd.DataFrame(output_data, columns=['id','context','penalty','violatelow'])
    # print(output_data)
    return output_data

if __name__ == '__main__':
    output_data = read_all_train()
    output_data.to_csv(pre_train_path,index=0)
    output_data.to_csv(pre_id_context_path, columns=['id','context'],index=0)
#    output_data.to_csv(pre_id_penalty_path, columns=['id','penalty'],index=0)
#    output_data.to_csv(pre_id_violatelow_path, columns=['id','violatelow'],index=0)
    # print(read_stop_words())