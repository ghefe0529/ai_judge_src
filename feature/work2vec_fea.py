# -*- coding: utf-8 -*-
# @Date    : 2018-06-12 16:37:46
# @Author  : GEFE (gh_efe@163.com)
# @Version : 1.0.0
# @Describe : 使用work2vec特征化数据

import numpy as np 
import pandas as pd 
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from collections import defaultdict

common_path = r'~/Documents/Study/Python/big_data/ai_judge'
# common_path = r'D:/gh/ai_judge'
# common_path = r'drive/Colab_Notebooks/ai_judge' 

word2Vec_model_path = common_path + r'/data/model/word2Vec_2.model'

id_context_path = common_path + r'/data/corpus/output/id_context.csv'
id_context_w2v_path = common_path + r'/data/feature/id_context_w2v.csv'

id_context_test_path = r'../../data/corpus/output/id_context_test.csv'
id_context_w2v_test_path = r'../../data/feature/id_context_w2v_test.csv'


def build_word2Vec_model(content):
    frequency = defaultdict(int)
    texts = content.apply(lambda x: x.split(' '))
    for text in texts:
        for word in text:
            frequency[word] += 1
    texts = texts.apply(lambda words: [word for word in words if frequency[word] > 5])

    print('--------------------')
 
    model = Word2Vec(texts, size=100, window=5, iter=15, workers=12)
    model.save(word2Vec_model_path)
    print('build model finish-------------------')

def word2Vec_handle_data(content, n):
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
    model = Word2Vec.load(word2Vec_model_path)
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
    pd.DataFrame(w2v_feat_avg[:n]).to_csv(id_context_w2v_path, header=None, index=0)
    pd.DataFrame(w2v_feat_avg[n:]).to_csv(id_context_w2v_test_path, header=None, index=0)

if __name__ == '__main__':
    content_train = pd.read_csv(id_context_path)['content']
    n = content_train.shape[0]
    content_text = pd.read_csv(id_context_test_path)['content']
    content = pd.concat([content_train, content_text], axis=0)
    build_word2Vec_model(content)
    word2Vec_handle_data(content, n)
    print("---------------------------------finish---------------------------------------")