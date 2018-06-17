# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd 

id_context_path = '../../data/corpus/output/id_context.csv'
only_context_path = '../../data/corpus/output/only_context.txt'

content = pd.read_csv(id_context_path)
content = content['context'].as_matrix()

f = open(only_context_path,'w',encoding='utf-8')
for doc in content:    
    f.write(doc+'\n')
f.close()