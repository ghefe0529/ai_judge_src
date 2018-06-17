# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

def content_to_txt(content, only_context_path):
    '''
    将content保存到txt中
    '''
    f = open(only_context_path,'w+',encoding='utf-8')
    for doc in content:    
        f.write(doc+'\n')
    f.close()

if __name__ == '__main__':
    
    '''
    将训练集中的content保存到txt中
    '''
    id_context_path = '../../data/corpus/output/id_content.csv'
    only_context_path = '../../data/corpus/output/only_content.txt'
    content = pd.read_csv(id_context_path)['content']
    content_to_txt(content, only_context_path)

    '''
    将测试集中的content保存到txt中
    '''
    id_context_test_path = '../../data/corpus/output/test.csv'
    content = pd.read_csv(id_context_test_path)['content']
    content_to_txt(content, only_context_path)