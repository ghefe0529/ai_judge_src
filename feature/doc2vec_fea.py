# -*- coding: utf-8 -*-
# @Date    : 2018-06-12 16:37:46
# @Author  : GEFE (gh_efe@163.com)
# @Version : 1.0.0
# @Describe : 使用doc2vec特征化数据

import numpy as np 
import pandas as pd 
from gensim.models import Doc2Vec
from collections import defaultdict

id_context_path = r'../../data/corpus/output/id_context.csv'
id_context_w2v_path = r'../../data/feature/id_context_d2v_2.csv'
word2Vec_model_path = r'../../data/model/doc2Vec_2.model'

def build_doc2Vec_model(content):
    frequency = defaultdict(int)
    texts = content.apply(lambda x: x.split(' '))
    # print(texts.values[1])
    for text in texts:
        for word in text:
            frequency[word] += 1
    texts = texts.apply(lambda words: [word for word in words if frequency[word] > 5])
    # print(texts)
    print('--------------------')
    # print(LineSentence(texts))
    model = Doc2Vec(texts, size=100, window=8, min_count=100, workers=8)
    model.save(word2Vec_model_path)
    print('build model finish-------------------')

def doc2Vec_handle_data(content):
    # 去掉单词低于5个后的向量
    frequency = defaultdict(int)
    texts = content.apply(lambda x: x.split(' '))
    # print(texts.values[1])
    for text in texts:
        for word in text:
            frequency[word] += 1
    texts = texts.apply(lambda words: [word for word in words if frequency[word] > 5])
    # print(texts.shape)

    vec_size = 100
    model = Doc2Vec.load(word2Vec_model_path)
    w2v_feat_avg = np.zeros((len(texts),vec_size))
    i = 0
    for words in texts:
        num = 0
        for word in words:
            vec = model[word]
            w2v_feat_avg[i, :] += vec
            num += 1
        w2v_feat_avg[i, :]= w2v_feat_avg[i, :]/num
        i += 1
        if i%1200 == 0:
            print(i)
    # pd.DataFrame(w2v_feat_avg).to_csv(id_context_w2v_path,index=0)
    print(pd.DataFrame(w2v_feat_avg))

if __name__ == '__main__':
    content = pd.read_csv(id_context_path,nrows=1000)['context']
    build_doc2Vec_model(content)
    # doc2Vec_handle_data(content)
    print("---------------------------------finish---------------------------------------")