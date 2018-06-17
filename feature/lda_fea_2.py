# -*- coding: utf-8 -*-
# @Date    : 2018-06-12 16:37:46
# @Author  : GEFE (gh_efe@163.com)
# @Version : 1.0.0
# @Describe : 使用lda特征化数据(one-hot特征化数据集)
# https://blog.csdn.net/real_myth/article/details/51239847

import numpy as np 
import pandas as pd 

import lda
from sklearn.feature_extraction.text import TfidfVectorizer


def tfidf_handle_data(content_data):
    tfv = TfidfVectorizer(min_df=3, max_df=0.95, sublinear_tf=True, stop_words='english')
    tfv.fit(content_data)
    result = tfv.transform(content_data)
    return result

def lda_handle_data(content):
    print('content shape is ',content.shape)
    lda_model = lda.LDA(n_topics=50, n_iter=500, random_state=1)
    lda_model.fit(content)
    return lda_model.doc_topic_

if __name__ == '__main__':
    id_context_path = '../../data/corpus/output/id_context.csv'
    id_context_lda_tow_path = r'../../data/feature/id_context_lda_tow.csv'
    content = pd.read_csv(id_context_path, nrows=100)['context']
    tfidf_data = tfidf_handle_data(content)
    doc_topic = lda_handle_data(tfidf_data)
    print(doc_topic.shape)
    
