import pandas as pd 
import jieba

common_path = r'../..'
common_path = r'D:/gh/ai_judge' 
common_path = r'drive/Colab_Notebooks/ai_judge' 

all_train_path = common_path + r'/data/corpus/input/train.txt'
all_test_path = common_path + r'/data/corpus/input/test.txt'
all_stop_path = common_path + r'/data/corpus/input/stop.txt'
pre_train_path = common_path + r'/data/corpus/output/train.csv'
pre_id_context_path = common_path + r'/data/corpus/output/id_content.csv'
pre_id_penalty_path = common_path + r'/data/corpus/output/id_penalty.csv'
pre_id_violatelow_path = common_path + r'/data/corpus/output/id_violatelow.csv'
test_path = common_path + r'/data/corpus/output/test.csv'

def read_stop_words():
    stop = []
    with open(all_stop_path, encoding='utf-8') as f:
        for line in f:
            stop.extend(line.split())
    return stop

def jieba_cut_text(text, stop):
    '''
    分词并根据stop文件剔除不必要的词组
    '''
    all_words = jieba.cut(text)
    useful_words = []
    for ele in all_words:
        # print(ele.split())
        if ele not in stop:
            useful_words.extend(ele.split())
    # print(useful_words)
    return ' '.join(useful_words)

def read_corpus_trian(trian_path,stop_words):
    '''
    读取并处理训练集的数据
    '''
    print("-----train is begin-----")
    return_data = []
    count = 0
    with open(trian_path, encoding='utf-8') as f:
        for line in f:
            line = line.replace('\n','')
            data = line.split('\t')
            tmp = dict()
            tmp['id']=data[0]
            tmp['content'] = jieba_cut_text(data[1], stop_words)
            tmp['penalty'] = data[2]
            tmp['violatelow'] = data[3].split(',')
            return_data.append(tmp)
            if count%1200 == 0:
               print('百分之%d' % ((count/120000)*100))
            count += 1

    return_data = pd.DataFrame(return_data, columns=['id','content','penalty','violatelow'])
    print("-----train is finish-----")
    return return_data

def read_corpus_test(test_path, stop):
    print("-----test is begin-----")
    return_data = []
    count = 0
    with open(test_path, encoding='utf-8') as f:
        for line in f:
            line = line.replace('\n','')
            data = line.split('\t')
            tmp = dict()
            tmp['id'] = data[0]
            tmp['content'] = jieba_cut_text(data[1], stop_words)
            return_data.append(tmp)
            if count%4000 == 0:
                print('百分之%d' % ((count/120000)*100))
            count += 1
    return_data = pd.DataFrame(return_data, columns=['id','content'])
    print("-----test is finish-----")
    return return_data

if __name__ == '__main__':
#   获取stop单词
    stop_words = read_stop_words()
#   获取训练集数据并存储
    output_train_data = read_corpus_trian(all_train_path,stop_words)
    output_train_data.to_csv(pre_train_path,index=0)
    output_train_data = pd.read_csv(pre_train_path)
    output_train_data.to_csv(pre_id_context_path, columns=['id','content'],index=0)
    output_train_data.to_csv(pre_id_penalty_path, columns=['id','penalty'],index=0)
    output_train_data.to_csv(pre_id_violatelow_path, columns=['id','violatelow'],index=0)
#   获取测试集数据并存储
    output_test_data = read_corpus_test(all_test_path, stop_words)
    output_test_data.to_csv(test_path,index=0,columns=['id','content'])