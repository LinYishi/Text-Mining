q# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 23:05:12 2018

@author: DELL
"""

##### DATA PRE PROCESS


import pickle
import pandas as pd
import numpy as np
articles = pd.DataFrame(pd.read_csv(r'C:\Users\DELL\Desktop\article_hexun.csv'))

'''
columns:
['author', 'title', 'cate_id', 'source', 'time', 'category', 'img_src',
       'objectId', 'body', 'createdAt', 'updatedAt', 'body_text', 'body_en',
       'tstamp', 'img_src_default', 'brief', 'title_en', 'cate_id2']
del articles['']...
'''
articles.fillna('')
f=lambda x:eval(x)
def f(x):
    try:
        y=eval(x)
    except TypeError:
        print(x)
        y=[]
    return y

articles['body1']=articles['body'].apply(f)

def g(x):
    try:
        text=x[0]
        tp=x[1]
        body=''
        for i in range(len(text)):
            if tp[i]=='text':
                body += text[i]
    except:
        body=''   
    return body
articles['body_text']=articles['body1'].apply(g)

del articles['body1']
f=open(r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\articles_hexun.pkl','wb')
pickle.dump(articles,f)
f.close()

f=open(r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\articles_hexun.pkl','rb')
data=pickle.load(f)
f.close()


def getlistnum(li):#这个函数就是要对列表的每个元素进行计数
    li = list(li)
    set1 = set(li)
    dict1 = {}
    for item in set1:
        dict1.update({item:li.count(item)})
    return dict1
 
zero_col_count = getlistnum(data.loc[data['body_text']!='']['category'])


def rm_claim(text):
    return re.sub('本文转自.*?，财经369.*?责任','',text)
articles_hexun['body_text_rmc']=articles_hexun['body_text'].apply(rm_claim)
articles_hexun['doc_length2'] = articles_hexun['body_text_rmc'].apply(lambda f:len(f))
articles_hexun=articles_hexun[articles_hexun['body_text']!='']
articles_hexun['doc_length2'].max()#19550
articles_hexun['doc_length2'].min()#7
articles_hexun['doc_length2'].mean()
def seg(text):
    seg_list=jieba.lcut(text)
    return seg_list
data['seg_list'] = data['body_text_rmc'].apply(seg)

stopwords=open(r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\stopwords1.txt','r',encoding='utf-8').readlines()
stopwords=[word.encode('utf-8').decode('utf-8-sig').strip() for word in stopwords]
stopwords_set=set(stopwords)

def rm_stpword(l1):
    return [x for x in l1 if x not in stopwords_set and x!='\n']

data['seg_list_rmstop'] = data['seg_list_rmstop'].apply(rm_stpword)

data['category']=data['category'].replace('港股','股票')

del data['body1']
del data['body_text_rmc']
del data['seg_list']

invalid_item = data[data['title'].isna()]['author'].values
data = data[-data['author'].isin(invalid_item)]

data = data[data['body_text']!='']

# 分train和test
import random
from collections import Counter
import math
random.seed(42)

def train_test_split(df, test_size=0.2):
    match_ids = df.index.unique().tolist()
    train_size = int(len(match_ids) * (1 - test_size))
    train_match_ids = random.sample(match_ids, train_size)

    train = df[df.index.isin(train_match_ids)]
    test = df[~df.index.isin(train_match_ids)]
    
    return train, test


def rm_space(lst):
    return [x.strip() for x in lst if ' ' not in x and '\n' not in x and '\t' not in x and x!='']

data['seg_list_rmstop'] = data['seg_list_rmstop'].apply(rm_space)

data['doc_word_only_list'] = data['seg_list_rmstop'].apply(lambda x: list(set(x))) # 每篇文章分词结果去重 - 这里不分train或test

data_train, data_test = train_test_split(data, test_size=0.2)


##********************************
## 预处理到这里
##********************************



# 计算idf
doc_num = data_train.shape[0]
word_list = data_train['doc_word_only_list'].as_matrix().tolist() # 将所有文章的所有单词（单篇去重）后合并为一个大list（里面是每个文章的小list）
word_list = [x for j in word_list for x in j] # 所有内容合并为一个大list #共3440622个词
word_counter = Counter(word_list) # 统计词频
word_list = list(word_counter.keys()) # 这里又一次把所有单词用列表形式展现 ，但应该是去重的，因为字典的key是unique的
word_count_list = list(word_counter.values()) # 对应地用列表存储了所有对应的词频
idf_list = [math.log(doc_num/x, 2) for x in word_count_list] # 计算idf
word_lib_df = pd.DataFrame({'word': word_list, 'idf': idf_list}) # 构建一个字段
word_lib_df.set_index('word', inplace=True) # 把word设置为index
word_lib_df['index'] = list(range(word_lib_df.shape[0]))
word_lib_route = r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib.csv'
word_lib_df.to_csv(word_lib_route)

word_lib_df['doc_num'] = 16039/(2**word_lib_df['idf'])

from collections import Counter
label_list = ['互联网金融', '保险', '债券', '基金', '外汇', '期货', '科技', '股票','银行','黄金']

for label in label_list:
    word_lib_df[label] = [0] * word_lib_df.shape[0]  # 初始化f(t, c)

group1 = data_train.groupby('category')

def get_word_count(lst):
    lst_all=[]
    for x in lst:
        lst_all.extend(x)
    word_counter = Counter(lst_all)
    return word_counter.items()
cate_word_count = group1['seg_list_rmstop'].agg(get_word_count)
#cate_word_count.index
#len(cate_word_count['保险'])
for label in label_list:
    dic = cate_word_count[label]
    for key,value in dic:
        word_lib_df.loc[key, label] = value
word_lib_df['total_freq'] = [0] * word_lib_df.shape[0]
for label in label_list:
    word_lib_df['total_freq'] += word_lib_df[label]
word_lib_route = r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib.csv'
word_lib_df.to_csv(word_lib_route)






###### 储存和加载data_train,data_test变量

f=open(r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\train_rmlowfreq.pkl','wb')
pickle.dump(data_train,f)
f.close()


f=open(r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\test_rmlowfreq.pkl','wb')
pickle.dump(data_test,f)
f.close()


f=open(r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\train_rmlowfreq.pkl','rb')
data_train = pickle.load(f)
f.close()

f=open(r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\test_rmlowfreq.pkl','rb')
data_test = pickle.load(f)
f.close()

f=open(r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\data.pkl','wb')
pickle.dump(data,f)
f.close()

f=open(r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\data_1216.pkl','rb')
data = pickle.load(f)
f.close()




with open(r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\train.txt','w',encoding='utf-8') as f:
    for i in range(data_train.shape[0]):
        categ = data_train.iloc[i,3]
        title = data_train.iloc[i,1]
        word_list = data_train.iloc[i,7]
        f.write(categ+'\t'+title+'\t'+' '.join(word_list)+'\n')

with open(r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\test.txt','w',encoding='utf-8') as f:
    for i in range(data_test.shape[0]):
        categ = data_test.iloc[i,3]
        title = data_test.iloc[i,1]
        word_list = data_test.iloc[i,7]
        f.write(categ+'\t'+title+'\t'+' '.join(word_list)+'\n')


### 构建DC相关元素的矩阵
import data_util as du
word_lib_df = du.get_word_lib_df()
word_lib_df['doc_num'] = 16039/(2**word_lib_df['idf'])

from collections import Counter
label_list = ['互联网金融', '保险', '债券', '基金', '外汇', '期货', '科技', '股票','银行','黄金']

for label in label_list:
    word_lib_df[label] = [0] * word_lib_df.shape[0]  # 初始化f(t, c)

group1 = data_train.groupby('category')

def get_word_count(lst):
    lst_all=[]
    for x in lst:
        lst_all.extend(x)
    word_counter = Counter(lst_all)
    return word_counter.items()
cate_word_count = group1['seg_list_rmstop'].agg(get_word_count)
cate_word_count.index
len(cate_word_count['保险'])
for label in label_list:
    dic = cate_word_count[label]
    for key,value in dic:
        word_lib_df.loc[key, label] = value
word_lib_df['total_freq'] = [0] * word_lib_df.shape[0]
for label in label_list:
    word_lib_df['total_freq'] += word_lib_df[label]
word_lib_route = r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib.csv'
word_lib_df.to_csv(word_lib_route)

def cal_leng(x):
    return len(x)
data_train['doc_length'] = data_train['seg_list_rmstop'].apply(cal_leng)
data_test['doc_length'] = data_test['seg_list_rmstop'].apply(cal_leng)
data_train.index=np.arange(data_train.shape[0])
data_test.index=np.arange(data_test.shape[0])

def default0(x):
    """Return f(), or 0 if it raises a KeyError."""
    try:
        return x*math.log(x, 2)
    except :
        return 0
import data_util as du
from imp import reload
reload(du)
word_lib_df = du.get_word_lib_df()
## 各个类别的频数
def getlistnum(li):#这个函数就是要对列表的每个元素进行计数
    li = list(li)
    set1 = set(li)
    dict1 = {}
    for item in set1:
        dict1.update({item:li.count(item)})
    return dict1
 
cate_count = getlistnum(data_train.loc[data_train['body_text']!='']['category_new'])
f=open(r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\cate_count.pkl','wb')
pickle.dump(cate_count,f)
f.close()
#word_lib_df['ftc_fc'] = [0]*word_lib_df.shape[0]
#for i in range(word_lib_df.shape[0]):
#    word_lib_df.loc[i,'ftc_fc'] = 


                    
                    
######################################################################
#   2018/12/16 去除数字/合并债券银行，去互金/去除手动挑出的停用词/去除低频词/重新构造word_lib_df
######################################################################

# 合并银行债券，去掉互金
def bind_cate(x):
    if x in ['银行','债券']:
        b= '银行债券'
    else:
        b= x
    return b
data['category_new'] = data['category'].apply(bind_cate)
new_data = data[data['category_new']!='互联网金融']
data = new_data
del new_data

# 重新去停用词
stopwords=open(r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\stopwords1.txt','r',encoding='utf-8').readlines()
stopwords=[word.strip() for word in stopwords]
stopwords_set=set(stopwords)

def rm_stpword(l1):
    return [x for x in l1 if x not in stopwords_set and x!='\n']

data['seg_list_rmstop'] = data['seg_list_rmstop'].apply(rm_stpword)

# 先存储一下，稍后引入进度条
import pickle
f=open(r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\data_1216.pkl','wb')
pickle.dump(data,f)
f.close()

from tqdm import tqdm
tqdm.pandas()

# 去除数字
import re
def rm_number(lst):
    return [x for x in lst if not re.findall('\d+\.?\d?\%?',x)]

data['seg_list_rmstop_rmnum'] = data['seg_list_rmstop'].apply(rm_number)

# 原来版本的word_lib_df中的低频词作参照，删低频词

data.index = np.arange(data.shape[0])
len_dict={}
for i in range(100):
    len_dict[str(i)] = len(data['seg_list_rmstop_rmnum'][i])

import data_util as du
word_lib_df = du.get_word_lib_df()                                  
low_freq_w = word_lib_df[word_lib_df['total_freq']<=2].index.tolist()
low_freq_w_set = set(low_freq_w)
#####  用了set,快了好几个量级
def rm_lowfreqword(l1):
    return [x for x in l1 if x not in low_freq_w_set and x!='\n']
data['seg_list_rmstop_rmnum'] = data['seg_list_rmstop_rmnum'].progress_apply(rm_lowfreqword)

len_dict1={}
for i in range(100):
    len_dict1[str(i)] = len(data['seg_list_rmstop_rmnum'][i])
    
# data的分词结果之前没处理干净,''还在，需要更新data，重新分train,test
def rm_space(lst):
    return [x.strip() for x in lst if ' ' not in x and '\n' not in x and '\t' not in x and x!='']
data['seg_list_rmstop_rmnum'] = data['seg_list_rmstop_rmnum'].apply(rm_space)




# 划分训练集测试集
import random
from collections import Counter
import math
random.seed(42)

def train_test_split(df, test_size=0.2):
    match_ids = df.index.unique().tolist()
    train_size = int(len(match_ids) * (1 - test_size))
    train_match_ids = random.sample(match_ids, train_size)

    train = df[df.index.isin(train_match_ids)]
    test = df[~df.index.isin(train_match_ids)]
    
    return train, test
data['seg_list_rmstop_rmnum']=data['seg_list_rmstop_rmnum'].apply(lambda x:x[0])
stopwords=open(r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\stopwords1.txt','r',encoding='utf-8').readlines()
stopwords=[word.encode('utf-8').decode('utf-8-sig').strip() for word in stopwords]
stopwords_set=set(stopwords)

def rm_stpword(l1):
    return [x for x in l1 if x not in stopwords_set and x!='\n']
data['seg_list_rmstop_rmnum']=data['seg_list_rmstop_rmnum'].apply(rm_stpword)
data['doc_word_only_list'] = data['seg_list_rmstop_rmnum'].apply(lambda x: list(set(x))) # 每篇文章分词结果去重 - 这里不分train或test
data_train, data_test = train_test_split(data, test_size=0.2)
def cal_leng(x):
    return len(x)
data_train['doc_length'] = data_train['seg_list_rmstop_rmnum'].apply(cal_leng)
data_test['doc_length'] = data_test['seg_list_rmstop_rmnum'].apply(cal_leng)
data_train.index=np.arange(data_train.shape[0])
data_test.index=np.arange(data_test.shape[0])

# 计算idf
doc_num = data_train.shape[0]
word_list = data_train['doc_word_only_list'].as_matrix().tolist() # 将所有文章的所有单词（单篇去重）后合并为一个大list（里面是每个文章的小list）
word_list = [x for j in word_list for x in j] # 所有内容合并为一个大list #共3440622个词
word_counter = Counter(word_list) # 统计词频
word_list = list(word_counter.keys()) # 这里又一次把所有单词用列表形式展现 ，但应该是去重的，因为字典的key是unique的
word_count_list = list(word_counter.values()) # 对应地用列表存储了所有对应的词频
idf_list = [math.log(doc_num/x, 2) for x in word_count_list] # 计算idf
word_lib_df = pd.DataFrame({'word': word_list, 'idf': idf_list,'doc_num':word_count_list}) # 构建一个字段
word_lib_df.set_index('word', inplace=True) # 把word设置为index
word_lib_df['index'] = list(range(word_lib_df.shape[0]))
#word_lib_df['doc_num'] = doc_num/(2**word_lib_df['idf']) #恢复文档数目


#统计分类别的词频
label_list = [ '保险', '银行债券', '基金', '外汇', '期货', '科技', '股票','黄金']

for label in label_list:
    word_lib_df[label] = [0] * word_lib_df.shape[0]  # 初始化f(t, c)

group1 = data_train.groupby('category_new')
def get_word_count(lst):
    lst_all=[]
    for x in lst:
        lst_all.extend(x)
    word_counter = Counter(lst_all)
    return word_counter.items()
cate_word_count = group1['seg_list_rmstop_rmnum'].agg(get_word_count) 
for label in label_list:
    dic = cate_word_count[label]
    for key,value in dic:
#        if key in data_train.loc[key,'title']:
#            word_lib_df.loc[key, label] = value*3  # 赋值f(t,c)，这步需要点时间，其实相当于把原来程序里的一部分不用重复的工作在外面做了
#        else:
        word_lib_df.loc[key, label] = value

word_lib_df['total_freq'] = [0] * word_lib_df.shape[0]
for label in label_list:
    word_lib_df['total_freq'] += word_lib_df[label] #计算总词频
word_lib_route = r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_rmlowfreq_1221.csv'
word_lib_df.to_csv(word_lib_route)


with open(r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\train_new.txt','w',encoding='utf-8') as f:
    for i in range(data_train.shape[0]):
        categ = data_train.iloc[i,-2]
        title = data_train.iloc[i,1]
        word_list = data_train.iloc[i,7]
        f.write(categ+'\t'+title+'\t'+' '.join(word_list)+'\n')

with open(r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\test_new.txt','w',encoding='utf-8') as f:
    for i in range(data_test.shape[0]):
        categ = data_test.iloc[i,-2]
        title = data_test.iloc[i,1]
        word_list = data_test.iloc[i,7]
        f.write(categ+'\t'+title+'\t'+' '.join(word_list)+'\n')

f=open(r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\train_rmlowfreq.pkl','wb')
pickle.dump(data_train,f)
f.close()


f=open(r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\test_rmlowfreq.pkl','wb')
pickle.dump(data_test,f)
f.close()

f=open(r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\data_1221.pkl','wb')
pickle.dump(data,f)
f.close()


## 有了data,一版word_lib_df，要去掉低频词
data.index = np.arange(data.shape[0])
word_lib_df = du.get_word_lib_df()                                  
low_freq_w = word_lib_df[word_lib_df['total_freq']<=2].index.tolist()
low_freq_w_set = set(low_freq_w)
#####  用了set,快了好几个量级
tqdm.pandas()
def rm_lowfreqword(l1):
    return [x for x in l1 if x not in low_freq_w_set and x!='\n']
data['seg_list_rmstop_rmnum'] = data['seg_list_rmstop_rmnum'].progress_apply(rm_lowfreqword)

data['doc_word_only_list'] = data['seg_list_rmstop_rmnum'].apply(lambda x: list(set(x))) # 每篇文章分词结果去重 - 这里不分train或test
data_train, data_test = train_test_split(data, test_size=0.2)
def cal_leng(x):
    return len(x)
data_train['doc_length'] = data_train['seg_list_rmstop_rmnum'].apply(cal_leng)
data_test['doc_length'] = data_test['seg_list_rmstop_rmnum'].apply(cal_leng)
data_train.index=np.arange(data_train.shape[0])
data_test.index=np.arange(data_test.shape[0])

# 计算idf
doc_num = data_train.shape[0]
word_list = data_train['doc_word_only_list'].as_matrix().tolist() # 将所有文章的所有单词（单篇去重）后合并为一个大list（里面是每个文章的小list）
word_list = [x for j in word_list for x in j] # 所有内容合并为一个大list #共3440622个词
word_counter = Counter(word_list) # 统计词频
word_list = list(word_counter.keys()) # 这里又一次把所有单词用列表形式展现 ，但应该是去重的，因为字典的key是unique的
word_count_list = list(word_counter.values()) # 对应地用列表存储了所有对应的词频
idf_list = [math.log(doc_num/x, 2) for x in word_count_list] # 计算idf
word_lib_df = pd.DataFrame({'word': word_list, 'idf': idf_list,'doc_num':word_count_list}) # 构建一个字段
word_lib_df.set_index('word', inplace=True) # 把word设置为index
word_lib_df['index'] = list(range(word_lib_df.shape[0]))
#word_lib_df['doc_num'] = doc_num/(2**word_lib_df['idf']) #恢复文档数目


#统计分类别的词频
label_list = [ '保险', '银行债券', '基金', '外汇', '期货', '科技', '股票','黄金']

for label in label_list:
    word_lib_df[label] = [0] * word_lib_df.shape[0]  # 初始化f(t, c)

group1 = data_train.groupby('category_new')
def get_word_count(lst):
    lst_all=[]
    for x in lst:
        lst_all.extend(x)
    word_counter = Counter(lst_all)
    return word_counter.items()
cate_word_count = group1['seg_list_rmstop_rmnum'].agg(get_word_count) 
for label in label_list:
    dic = cate_word_count[label]
    for key,value in dic:
#        if key in data_train.loc[key,'title']:
#            word_lib_df.loc[key, label] = value*3  # 赋值f(t,c)，这步需要点时间，其实相当于把原来程序里的一部分不用重复的工作在外面做了
#        else:
        word_lib_df.loc[key, label] = value

word_lib_df['total_freq'] = [0] * word_lib_df.shape[0]
for label in label_list:
    word_lib_df['total_freq'] += word_lib_df[label] #计算总词频
word_lib_route = r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_rmlowfreq_1221.csv'
word_lib_df.to_csv(word_lib_route)

with open(r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\train_new.txt','w',encoding='utf-8') as f:
    for i in range(data_train.shape[0]):
        categ = data_train.iloc[i,-2]
        title = data_train.iloc[i,1]
        word_list = data_train.iloc[i,7]
        f.write(categ+'\t'+title+'\t'+' '.join(word_list)+'\n')

with open(r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\test_new.txt','w',encoding='utf-8') as f:
    for i in range(data_test.shape[0]):
        categ = data_test.iloc[i,-2]
        title = data_test.iloc[i,1]
        word_list = data_test.iloc[i,7]
        f.write(categ+'\t'+title+'\t'+' '.join(word_list)+'\n')

f=open(r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\train_rmlowfreq.pkl','wb')
pickle.dump(data_train,f)
f.close()


f=open(r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\test_rmlowfreq.pkl','wb')
pickle.dump(data_test,f)
f.close()

f=open(r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\data_1221.pkl','wb')
pickle.dump(data,f)
f.close()







# 计算idf
doc_num = data_train.shape[0]
word_list = data_train['doc_word_only_list'].as_matrix().tolist() # 将所有文章的所有单词（单篇去重）后合并为一个大list（里面是每个文章的小list）
word_list = [x for j in word_list for x in j] # 所有内容合并为一个大list #共3440622个词
word_counter = Counter(word_list) # 统计词频
word_list = list(word_counter.keys()) # 这里又一次把所有单词用列表形式展现 ，但应该是去重的，因为字典的key是unique的
word_count_list = list(word_counter.values()) # 对应地用列表存储了所有对应的词频
idf_list = [math.log(doc_num/x, 2) for x in word_count_list] # 计算idf
word_lib_df = pd.DataFrame({'word': word_list, 'idf': idf_list}) # 构建一个字段
word_lib_df.set_index('word', inplace=True) # 把word设置为index
word_lib_df['index'] = list(range(word_lib_df.shape[0]))
word_lib_df['doc_num'] = 16039/(2**word_lib_df['idf']) #恢复文档数目


#统计分类别的词频
label_list = [ '保险', '银行债券', '基金', '外汇', '期货', '科技', '股票','黄金']

for label in label_list:
    word_lib_df[label] = [0] * word_lib_df.shape[0]  # 初始化f(t, c)

group1 = data_train.groupby('category_new')
def get_word_count(lst):
    lst_all=[]
    for x in lst:
        lst_all.extend(x)
    word_counter = Counter(lst_all)
    return word_counter.items()
cate_word_count = group1['seg_list_rmstop_rmnum'].agg(get_word_count) 
for label in label_list:
    dic = cate_word_count[label]
    for key,value in dic:
        if key in data_train.loc[key,'title']:
            word_lib_df.loc[key, label] = value*3  # 赋值f(t,c)，这步需要点时间，其实相当于把原来程序里的一部分不用重复的工作在外面做了
        else:
            word_lib_df.loc[key, label] = value

word_lib_df['total_freq'] = [0] * word_lib_df.shape[0]
for label in label_list:
    word_lib_df['total_freq'] += word_lib_df[label] #计算总词频
word_lib_route = r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_rmlowfreq.csv'
word_lib_df.to_csv(word_lib_route)



#### bdc title word_lib_df
import data_util as du
import imp
imp.reload(du)
word_lib_df = du.get_word_lib_df()
data_train
def in_title(word_list,title):
    intitle_words=[]
    for word in set(word_list):
        if word in title:
            intitle_words.append(word)
    return set(intitle_words)
word_lib_df_bdctitle2 = word_lib_df.copy()
data_train['intitle_words'] = data_train.apply(lambda y:in_title(y['seg_list_rmstop_rmnum'],y['title']),axis=1)
from tqdm import tqdm
for i in tqdm(range(data_train.shape[0])):
    for word in data_train.loc[i,'intitle_words']:
        word_lib_df_bdctitle2.loc[word,data_train.loc[i,'category_new']]+=1

word_lib_bdctitle_route = r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_df_bdctitle2.csv'
word_lib_df_bdctitle2.to_csv(word_lib_bdctitle_route)

word_lib_df_abct3 = word_lib_df.copy()                
def in_abstract(word_list,perc=0.1):
    return set(word_list[:int(len(word_list)*perc)])
data_train['inabst_words']=data_train['seg_list_rmstop_rmnum'].apply(in_abstract)
for i in tqdm(range(data_train.shape[0])):
    for word in data_train.loc[i,'inabst_words']:
        word_lib_df_abct3.loc[word,data_train.loc[i,'category_new']]+=2
word_lib_bdctitle_route = r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_df_abct3.csv'
word_lib_df_abct3.to_csv(word_lib_bdctitle_route)