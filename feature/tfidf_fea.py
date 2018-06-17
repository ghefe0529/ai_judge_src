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

id_context_path = '../../data/corpus/output/id_context.csv'
id_penalty_path = '../../data/corpus/output/id_penalty.csv'
fea_id_context_tfidf_loass_path = '../../data/feature/id_context_tfidf_loass.csv'
fea_id_context_tfidf_logistic_path = '../../data/feature/id_context_tfidf_logistic.csv'


def select_coef_by_model(coef, data, min=0.01):
    del_weight = []
    # print(type(coef))
    # print(len(coef))
    for ele,index in zip(coef, range(len(coef))):
        if np.abs(ele) < min:
            del_weight.append(index)
    # print('del_weight type is ',type(del_weight))
    # print('del_weight len is ',len(del_weight))
    # print('del_weight[1] is ',del_weight[1])
    # print('data len is ', data.shape)
    data = np.delete(data, del_weight, axis=1)
    # print('data len is ', data.shape)
    return data

def tfidf_handle_data(content_data):
    tfv = TfidfVectorizer(min_df=3, max_df=0.95, sublinear_tf=True, stop_words='english')
    tfv.fit(content_data)
    result = tfv.transform(content_data)
    return result

if __name__ == '__main__':
    content_data = pd.read_csv(id_context_path)
    tfidf_data = tfidf_handle_data(content_data['context'])
    # tfidf_data = pd.DataFrame(tfidf_data.todense())
    # tfidf_data.to_csv(fea_id_context_path,index=0)
    Y = pd.read_csv(id_penalty_path)['penalty']
    x_train, x_test, y_train, y_test = train_test_split(tfidf_data, Y, test_size=.2, random_state=0)
    # print('y_test', len(y_test))
    # print('x_train', x_train)
    # print('y_train', y_train)
    '''
    逻辑回归训练模型降维
    '''

    # print(tfidf_data.shape)
    # lgr = LogisticRegression()
    # lgr.fit(tfidf_data,Y)
    # # y_pre = lgr.predict(x_test)
    # tfidf_new_data = lgr.predict_proba(tfidf_data)
    # print(tfidf_new_data.shape)
    # x_new_train = pd.DataFrame(tfidf_new_data)
    # x_new_train.to_csv(fea_id_context_tfidf_logistic_path)
    
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
    # lsvc = LinearSVC(C=0.01, penalty='l1',dual=False).fit(x_train, y_train)
    lasso = Lasso(alpha=0.0001)
    lasso.fit(tfidf_data, Y)
    # print(len(lasso.coef_))
    # model = SelectFromModel(lsvc, prefit=True)
    tfidf_new = select_coef_by_model(lasso.coef_, tfidf_data.todense())
    
    # model = SelectFromModel(lasso, prefit=True)
    # tfidf_new = model.transform(tfidf_data)
    
    print(tfidf_new.shape)
    print(type(tfidf_new))
    tfidf_new = pd.DataFrame(tfidf_new)
    print(tfidf_new.shape)
    tfidf_new.to_csv(fea_id_context_tfidf_loass_path)