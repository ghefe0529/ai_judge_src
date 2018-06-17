# -*- coding: utf-8 -*-

import pandas as pd 
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.models.tfidfmodel import TfidfModel
import numpy as np

common_path = r'../../'

id_content_path = common_path + r'data/corpus/output/id_context.csv'
id_context_lda_path = common_path + r'data/feature/id_context_lda.csv'
def get_lda_feacture_name(size):
    names = []
    for i in range(size):
        names.append('lda'+str(i))
    return names

def get_lda_feature():
    doc_train = pd.read_csv(id_content_path)
    documents = doc_train['context'].apply(lambda x: x.split(' '))
#    建立词和ID的映射字典(id:word)
    dictionary = corpora.Dictionary(documents)
#    建立文档和id和list(tuple(id,num)) of list df
    ds_df = [dictionary.doc2bow(document) for document in documents]
#    建立tfidf模型，通过语料文档的tf，预测的时候只要提供语料的df
    tfidf_model = TfidfModel(ds_df)
#    获取文档的tdf获取文档tfidf
    ds_tfidf = tfidf_model[ds_df]
#    定义文档的主题个数
    n = 50
#    构建lda模型，输入参数是文档的tfidf，并指明主题的个数
    lda_model = LdaModel(ds_tfidf, num_topics=n, passes=10, random_state=12)
    vec_size = (len(documents),n)
    lda_feature = np.zeros(vec_size)
    i = 0
    
    for doc in ds_tfidf:
        topics = lda_model.get_document_topics(doc, minimum_probability=0.01)
        for topic in topics:
            num_topic = topic[0]
            prob = round(topic[1],5)
            lda_feature[i, num_topic] = prob
        i += 1
        
    f_names = get_lda_feacture_name(n)
    pd.DataFrame(lda_feature, columns=f_names).to_csv(id_context_lda_path, index=0)
    
    
get_lda_feature()