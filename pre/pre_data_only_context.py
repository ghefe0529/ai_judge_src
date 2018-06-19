# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

def content_to_txt(content, only_context_path):
    '''
    将content保存到txt中
    '''
    f = open(only_context_path,'w',encoding='utf-8')
    for doc in content:    
        f.write(doc+'\n')
    f.close()

if __name__ == '__main__':
    
    '''
    将训练集中的content保存到txt中
    '''
    
    # common_path = r'~/Documents/Study/Python/big_data/ai_judge/'
    common_path = r'D:/gh'
    # common_path = r'drive/Colab_Notebooks/ai_judge' 
    id_context_path = common_path + r'/data/corpus/output/id_content.csv'
    only_context_path = common_path + r'/data/corpus/output/only_content.txt'
    content_train = pd.read_csv(id_context_path)['content']

    '''
    将测试集中的content保存到txt中
    '''
    id_context_test_path = common_path + r'/data/corpus/output/test.csv'
    content_test = pd.read_csv(id_context_test_path)['content']
    '''
    合并测试集和训练集
    '''
    print('content_train is ',content_train.shape)
    print('content_test is ',content_test.shape)
    
    content = pd.concat([content_train, content_test],axis=0)
    print('content_train is ',content.shape)
    content_to_txt(content, only_context_path)