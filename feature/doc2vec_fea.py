# -*- coding: utf-8 -*-
# @Date    : 2018-06-12 16:37:46
# @Author  : GEFE (gh_efe@163.com)
# @Version : 1.0.0
# @Describe : 使用doc2vec特征化数据

import numpy as np 
import pandas as pd 
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedLineDocument
from collections import defaultdict

common_path = r'../..'
common_path = r'D:/gh/ai_judge' 
common_path = r'drive/Colab_Notebooks/ai_judge' 

only_context_path = common_path + r'/data/corpus/output/only_context.txt'
only_context_test_path = common_path + r'/data/corpus/output/only_context_test.txt'

doc2Vec_model_path = common_path + r'/data/model/doc2Vec.model'

id_context_path = common_path + r'/data/corpus/output/id_context.csv'
id_context_d2v_path = common_path + r'/data/feature/id_context_d2v.csv'
id_context_d2v_doc_path = common_path + r'/data/feature/id_context_d2v_doc.csv'

id_context_test_path = common_path + r'/data/corpus/output/id_context_test.csv'
id_context_d2v_test_path = common_path + r'/data/feature/id_context_d2v_test.csv'
id_context_d2v_doc_test_path = common_path + r'/data/feature/id_context_d2v_doc_test.csv'

def build_doc2Vec_model():
    '''
    建立doc2Vec模型
    '''
    texts = TaggedLineDocument(only_context_path)
    print(len(texts))

    print('----------doc2Vec model is building----------')
    # print('texts type is ',type(texts))

    model = Doc2Vec(texts,vector_size=100, window=8, min_count=5, workers=5)
    model.save(doc2Vec_model_path)
    print('-----------build model is finish-------------')

def doc2Vec_handle_data(content, save_path):
    '''
    去掉词组数低于5个的词组
    再计算一个文档内所有词组向量和的平均值
    '''
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

    model = Doc2Vec.load(doc2Vec_model_path)
    # 计算所有词组向量和的平均值
    d2v_feat_avg = np.zeros((len(texts),vec_size))
    i = 0
    for words in texts:
        num = 0
        for word in words:
            vec = model[word]
            d2v_feat_avg[i, :] += vec
            num += 1
        d2v_feat_avg[i, :]= d2v_feat_avg[i, :]/num
        i += 1
        if i%1200 == 0:
            print(i)
    pd.DataFrame(d2v_feat_avg).to_csv(save_path, index=0)

def doc2Vec_handle_data_two(size_begin, size_end, save_path):
    '''
    直接获取每个文档的向量
    '''
    model = Doc2Vec.load(doc2Vec_model_path)
#    print(model.docvecs[0])
    d2v_feat = list()
    for i in range(len(size_begin,size_end)):
        d2v_feat.append(model.docvecs[i])
    pd.DataFrame(d2v_feat).to_csv(save_path,index=0)

if __name__ == '__main__':
    # 训练集
    content = pd.read_csv(id_context_path)['context']
    build_doc2Vec_model()
    doc2Vec_handle_data(content, id_context_d2v_path)
    doc2Vec_handle_data_two(0, len(content),id_context_d2v_doc_path)
    # 测试集
    content_test = pd.read_csv(id_context_test_path)['context']
    build_doc2Vec_model()
    doc2Vec_handle_data(content, id_context_d2v_test_path)
    doc2Vec_handle_data_two(len(content),len(content_test),id_context_d2v_doc_test_path)
    print("---------------------------------finish---------------------------------------")