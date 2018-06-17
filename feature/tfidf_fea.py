# -*- coding: utf-8 -*-
# @Date    : 2018-06-12 11:49:47
# @Author  : GEFE (gh_efe@163.com)
# @Version : 1.0.0
# @Describe : 将词组特征为tdidf形式

import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import  GaussianNB,MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,Lasso,LinearRegression
from sklearn.feature_selection import RFE

common_path = r'../..'
common_path = r'D:/gh/ai_judge'

id_context_path = common_path + r'/data/corpus/output/id_content.csv'
id_penalty_path = common_path + r'/data/corpus/output/id_penalty.csv'
fea_id_context_tfidf_loass_path = common_path + r'/data/feature/id_context_tfidf_loass.csv'
fea_id_context_tfidf_logistic_path = common_path + r'/data/feature/id_context_tfidf_logistic.csv'


def select_coef_by_model(coef, data, min=0.01):
    del_weight = []
    for ele,index in zip(coef, range(len(coef))):
        if np.abs(ele) < min:
            del_weight.append(index)
    data = np.delete(data, del_weight, axis=1)
    return data

def tfidf_handle_data(data):
    tfv = TfidfVectorizer(min_df=3, max_df=0.95, sublinear_tf=True, stop_words='english')
    tfv.fit(data)
    result = tfv.transform(data)
    return result

if __name__ == '__main__':
    # 将分词后的content用tfidf形式保存
    content_data = pd.read_csv(id_context_path)['content']
    tfidf_data = tfidf_handle_data(content_data)
    Y = pd.read_csv(id_penalty_path)['penalty']

    
    '''
    使用逻辑回归训练模型对tfidf的特征降维
    '''
    print(tfidf_data.shape)
    lgr = LogisticRegression()
    lgr.fit(tfidf_data,Y)
    # y_pre = lgr.predict(x_test)
    tfidf_new_data = lgr.predict_proba(tfidf_data)
    print(tfidf_new_data.shape)
    x_new_train = pd.DataFrame(tfidf_new_data)

    x_new_train.to_csv(fea_id_context_tfidf_logistic_path, header=None, index=0)
    
    '''
    REF特征选择算法
    '''
    # lr = LinearRegression()
    # # lgr = Lasso(alpha=0.001)
    # rfe = RFE(lr, n_features_to_select=1)
    # rfe.fit(x_train, y_train)
    # print(len(rfe.ranking_))
    '''
    基于L1的特征选择
    '''


    # from sklearn.svm import LinearSVC
    from sklearn.linear_model import Lasso
    from sklearn.datasets import load_iris
    from sklearn.feature_selection import SelectFromModel

    print(tfidf_data.shape)
    lasso = Lasso(alpha=0.0001)
    lasso.fit(tfidf_data, Y)
    tfidf_new = select_coef_by_model(lasso.coef_, tfidf_data.todense())

    print(tfidf_new.shape)
    print(type(tfidf_new))
    tfidf_new = pd.DataFrame(tfidf_new)

    tfidf_new.to_csv(fea_id_context_tfidf_loass_path, header=None, index=0)