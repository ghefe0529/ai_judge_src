# -*- coding: utf-8 -*-
# @Date    : 2018-06-17 09:41:47
# @Author  : GEFE (gh_efe@163.com)
# @Version : 1.0.0
# @Describe : 训练模型

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

def build_model(model_name, X, Y):
    print('now build model is ',model_name)
    print('x shape is ', X.shape)
    print('y shape is ', Y.shape)
    # print('y[0] is ', Y[0])
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    lgr = model_name
    lgr.fit(x_train, y_train)
    y_pre = lgr.predict(x_test)
    erro_num = 0
    print(erro_num, len(y_test))
    for test,pre in zip(y_test, y_pre):
        if test != pre:
            erro_num += 1
    print('错误率为 %.3f' % (erro_num/len(y_test)))
    
if __name__ == '__main__':
    common_path = r'~/Documents/Study/Python/big_data/ai_judge'

    id_context_d2v_path = common_path + r'/data/feature/id_context_d2v.csv'
    id_context_w2v_path = common_path + r'/data/feature/id_context_w2v.csv'
    id_context_lda_path = common_path + r'/data/feature/id_context_lda.csv'
    id_context_loass_path = common_path + r'/data/feature/id_context_tfidf_loass.csv'
    id_penalty_path = common_path + r'/data/corpus/output/id_penalty.csv'

    # 基于doc2vec特征
    print('基于doc2vec特征')
    train_X_data_by_d2v = pd.read_csv(id_context_d2v_path)
    train_Y_data = pd.read_csv(id_penalty_path)['penalty']
    # build_model(GaussianNB, train_X_data_by_d2v, train_Y_data)
    # 基于work2vec特征
    print('基于work2vec特征')
    train_X_data_by_w2v = pd.read_csv(id_context_w2v_path)
    # build_model(GaussianNB, train_X_data_by_w2v, train_Y_data)
    # 基于lda特征
    print('基于lda特征')
    train_X_data_by_lda = pd.read_csv(id_context_lda_path)
    # build_model(GaussianNB, train_X_data_by_lda, train_Y_data)

    # tfidf-lasso 特征
    train_X_data_by_loass = pd.read_csv(id_context_loass_path).as_matrix()[:, 1:]
    build_model(DecisionTreeClassifier(), train_X_data_by_loass, train_Y_data)

    # 基于多个特征
    # print(type(train_X_data_by_d2v),type(train_X_data_by_w2v))
    # train_X_data_by_d2v_and_w2v = pd.concat([train_X_data_by_w2v, train_X_data_by_d2v, train_X_data_by_lda], axis=1)
    # build_model(RandomForestRegressor(), train_X_data_by_d2v_and_w2v, train_Y_data)