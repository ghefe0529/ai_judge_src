# -*- coding: utf-8 -*-
# @Date    : 2018-06-12 17:06:23
# @Author  : GEFE (gh_efe@163.com)
# @Version : 1.0.0
# @Describe : 根据特征使用贝叶斯训练模型

import numpy as np 
import pandas as pd 
from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

common_path = r'~/Documents/Study/Python/big_data/ai_judge'

id_context_w2v_path = common_path + r'/data/feature/id_context_w2v.csv'
id_penalty_path = common_path + r'/data/corpus/output/id_penalty.csv'

X = pd.read_csv(id_context_w2v_path)
# print(X)
# scaler = StandardScaler()
# X = scaler.fit_transform(X)
# pca = PCA(n_components=100)
# X = pca.fit_transform(X)
# print(X)
Y = pd.read_csv(id_penalty_path)['penalty']
# print(Y)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.2, random_state=0)

m = GaussianNB()
m.fit(x_train, y_train)
y_pre = m.predict(x_test)
erro_count = 0
for y1,y2 in zip(y_pre,y_test):
    if y1 == y2:
        continue
    erro_count += 1

print("错误率%.3f" % (erro_count/len(y_test)))

