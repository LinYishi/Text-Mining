


#########
#Algorithm
#########

from collections import Counter
import math
import data_util as du
import imp
imp.reload(du)
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import pandas as pd
import pickle
f=open(r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\cate_count.pkl','rb')
cate_count = pickle.load(f)
f.close()

def tf_idf(file_route):
    '''
    KNN:
    正确率：0.70
    Macro_F1: 0.6809592503036148
    Micro_F1: 0.7834627683873915
    SVM:
    Macro_F1: 0.3034147735580599
    Micro_F1: 0.7880310644129741
    :param file_route:
    :return:
    '''
    data = du.get_data(file_route)
    doc_list = data.split('\n')[:-1]

    doc_num = len(doc_list)
    word_lib_df = du.get_word_lib_df()
    word_lib_num = word_lib_df.shape[0]

    label_list = []
    tf_idf = []
    col = []
    row = []
    doc_no = 0

    for doc in tqdm(doc_list):
        label_list.append(doc.split('\t')[0])
        doc_word_list = doc.split('\t')[2].strip().split(' ')
        doc_word_counter = Counter(doc_word_list)
        doc_lenth = len(doc_word_list)
        for word, count in doc_word_counter.items():
            try:
                one_word_tf = count/doc_lenth
                one_word_idf = word_lib_df.loc[word, 'idf']  # 对于测试集有可能出现异常，即测试集中的某个词在词库中查不到因为词库是训练集构造的
                # fixme 用训练集构造的词库中报错说没有训练集第4543个doc中的nan这个词而实际上词库中有这个词 KeyError: 'the label [nan] is not in the [index]'
                one_word_tf_idf = one_word_tf*one_word_idf
                row.append(doc_no)
                col.append(word_lib_df.loc[word, 'index'])
                tf_idf.append(one_word_tf_idf)
            except:
                continue
        doc_no += 1
    tf_idf_train_feature_sparse_matrix = csr_matrix((tf_idf, (row, col)), shape=(doc_num, word_lib_num))
    # print(len(train_label_list), train_label_list[0])
    le = LabelEncoder()
    le.fit(label_list)
    # print(le.classes_)  # 显示出共八种标签['acq' 'crude' 'earn' 'grain' 'interest' 'money-fx' 'ship' 'trade']
    train_label_list = list(le.transform(label_list))
    # print(train_label_list[0: 5])  # 显示训练集中前五个文档的被编码后的标签[2, 0, 2, 2, 2]
    return tf_idf_train_feature_sparse_matrix, train_label_list


def tf_dc(data0,title_times=1,abst_times=1,perc=0,word_lib_path=r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_rmlowfreq_1221.csv'):
    word_lib_df = du.get_word_lib_df(word_lib_path)
    label_list = [ '保险', '基金', '外汇', '期货', '科技', '股票','银行债券','黄金']
    doc_label_list = []
    row = []
    col = []
    value = []
    doc_no = 0
    for i in tqdm(range(data0.shape[0])):
#        title = data0.loc[i,'title']
        temp1 = word_lib_df.loc[data0.loc[i,'seg_list_rmstop_rmnum']]
        temp1 = temp1[temp1['index']==temp1['index']]  ## 去除测试集中新的词，新的词index是nan，nan!=nan
        doc_label_list.append(data0.loc[i,'category_new'])        
        tmp_cnt = temp1.index.value_counts() 
#        abst_words = set(data0.loc[i,'seg_list_rmstop_rmnum'][:int(perc*temp1.shape[0])])
        temp1 = pd.concat([temp1.drop_duplicates(),tmp_cnt],axis=1)
        temp1['one_word_tf'] = temp1['word']/temp1['word'].sum()
#        if abst_times!=1 or title_times!=1:
#            for word in temp1.index:
#                if word in title:
#                    temp1.loc[word,'one_word_tf'] = temp1.loc[word,'one_word_tf']*abst_times
#                if word in abst_words:
#                    temp1.loc[word,'one_word_tf'] = temp1.loc[word,'one_word_tf']*title_times
        temp1['word_dc'] = [0]*temp1.shape[0]
        for label in label_list:
    #        temp1[label+'_pec'] = [0]*temp1.shape[0]
            temp1[label+'_pec'] = temp1[label]/temp1['doc_num']
            temp1[label+'_dc'] = temp1[label+'_pec'].apply(default0)
            temp1['word_dc'] = temp1['word_dc'] + temp1[label+'_dc']
        temp1['word_dc_final'] = [1]*temp1.shape[0] + temp1['word_dc']/math.log(len(label_list), 2)
    #        temp1[temp1[label]!=0][label+'_pec'] = temp1[temp1[label]!=0][label]/temp1[temp1[label]!=0]['doc_num']
    #        temp1[temp1[label]!=0][label+'_dc'] = temp1[temp1[label]!=0][label+'_pec'].apply(default0)
        temp1['tf_dc'] = temp1['word_dc_final']*temp1['one_word_tf']
        col_list = temp1['index'].tolist()
        value_list = temp1['tf_dc'].tolist()
    #    doc_no = doc_no+temp1.shape[0]
        row_list = [doc_no]*temp1.shape[0]
        row.extend(row_list)
        col.extend(col_list)
        value.extend(value_list)
        doc_no += 1
        
    tf_dc_feature_sparse_matrix = csr_matrix((value, (row, col)), shape=(len(doc_label_list), word_lib_df.shape[0]))
    le = LabelEncoder()
    le.fit(doc_label_list)
    return tf_dc_feature_sparse_matrix, le.transform(doc_label_list)


def tf_bdc(data0,title_times=1,abst_times=1,perc=0,word_lib_path=r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_rmlowfreq_1221.csv'):
    word_lib_df = du.get_word_lib_df(word_lib_path)
    label_list = [ '保险', '银行债券', '基金', '外汇', '期货', '科技', '股票','黄金']
    doc_label_list = []
    row = []
    col = []
    value = []
    doc_no = 0
    for i in tqdm(range(data0.shape[0])):
#        title = data0.loc[i,'title']
        temp1 = word_lib_df.loc[data0.loc[i,'seg_list_rmstop_rmnum']]
        temp1 = temp1[temp1['index']==temp1['index']]  ## 去除测试集中新的词，新的词index是nan，nan!=nan
        doc_label_list.append(data0.loc[i,'category_new'])        
        tmp_cnt = temp1.index.value_counts()
#        for word in tmp_cnt.index:
#            if word in title:
#                tmp_cnt[word] = tmp_cnt[word]*title_times
#        abst_words = set(data0.loc[i,'seg_list_rmstop_rmnum'][:int(perc*temp1.shape[0])])
        temp1 = pd.concat([temp1.drop_duplicates(),tmp_cnt],axis=1)       
        temp1['one_word_tf'] = temp1['word']/temp1['word'].sum()
#        if abst_times!=1 or title_times!=1:
#            for word in temp1.index:
#                if word in title:
#                    temp1.loc[word,'one_word_tf'] = temp1.loc[word,'one_word_tf']*title_times
#                if word in abst_words:
#                    temp1.loc[word,'one_word_tf'] = temp1.loc[word,'one_word_tf']*abst_times
        temp1['word_dc'] = [0]*temp1.shape[0]
        temp1['word_dc_pec'] = [0]*temp1.shape[0]
        for label in label_list:
    #        temp1[label+'_pec'] = [0]*temp1.shape[0]
            temp1[label+'_pec'] = temp1[label]/cate_count[label]
            temp1['word_dc_pec'] = temp1['word_dc_pec'] + temp1[label+'_pec']
        for label in label_list:
#            temp1[label+'_pec'] = temp1[label+'_pec']/temp1[label+'_pec'].sum()
            temp1[label+'_pec1'] = temp1[label+'_pec'] / temp1['word_dc_pec']
            temp1[label+'_dc'] = temp1[label+'_pec1'].apply(default0)
            temp1['word_dc'] = temp1['word_dc'] + temp1[label+'_dc']
        temp1['word_dc_final'] = [1]*temp1.shape[0] + temp1['word_dc']/math.log(len(label_list), 2)
    #        temp1[temp1[label]!=0][label+'_pec'] = temp1[temp1[label]!=0][label]/temp1[temp1[label]!=0]['doc_num']
    #        temp1[temp1[label]!=0][label+'_dc'] = temp1[temp1[label]!=0][label+'_pec'].apply(default0)
        temp1['tf_bdc'] = temp1['word_dc_final']*temp1['one_word_tf']
#        temp1['idf_bdc'] = temp1['word_dc_final']*temp1['idf']
        col_list = temp1['index'].tolist()
        value_list = temp1['tf_bdc'].tolist()
    #    doc_no = doc_no+temp1.shape[0]
        row_list = [doc_no]*temp1.shape[0]
        row.extend(row_list)
        col.extend(col_list)
        value.extend(value_list)
        doc_no += 1       
    tf_dc_feature_sparse_matrix = csr_matrix((value, (row, col)), shape=(len(doc_label_list), word_lib_df.shape[0]))
    le = LabelEncoder()
    le.fit(doc_label_list)
    return tf_dc_feature_sparse_matrix, le.transform(doc_label_list)


def default0(x):
    try:
        return x*math.log(x, 2)
    except :
        return 0
    
def tfc(file_route,title_times=1,abst_times=1,perc=0,word_lib_path=r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_rmlowfreq_1221.csv'):
    data = du.get_data(file_route)
    doc_list = data.split('\n')[:-1]
    doc_num = len(doc_list)
    word_lib_df = du.get_word_lib_df(word_lib_path)
    word_lib_num = word_lib_df.shape[0]
    label_list = []
    tfc = []
    col = []
    row = []
    doc_no = 0
    for doc in tqdm(doc_list):
        label_list.append(doc.split('\t')[0])
        doc_word_list = doc.split('\t')[2].strip().split(' ')
        title = doc.split('\t')[1]
        doc_word_counter = Counter(doc_word_list)
        doc_lenth = len(doc_word_list)
        abst_words=set(doc_word_list[:int(doc_lenth*perc)])
        tmp_tf_idf_list=[]
        for word, count in doc_word_counter.items():
            try:
                if word in title:
                    count = count*title_times
                if perc!=0 and word in abst_words:
                    count = count*abst_times
                one_word_tf = count/doc_lenth
                one_word_idf = word_lib_df.loc[word, 'idf'] 
                one_word_tf_idf = one_word_tf*one_word_idf
                row.append(doc_no)
                col.append(word_lib_df.loc[word, 'index'])
                tmp_tf_idf_list.append(one_word_tf_idf)
            except:
                continue
        fenzi = math.sqrt(sum([x**2 for x in tmp_tf_idf_list]))
        tmp_tf_idf = [x/fenzi for x in tmp_tf_idf_list] ## 化为单位向量
        tfc.extend(tmp_tf_idf)
        doc_no += 1    
    tfc_train_feature_sparse_matrix = csr_matrix((tfc, (row, col)), shape=(doc_num, word_lib_num))
    le = LabelEncoder()
    le.fit(label_list)
    # print(le.classes_)
    train_label_list = list(le.transform(label_list))
    return tfc_train_feature_sparse_matrix, train_label_list


def itc(file_route,title_times=1,abst_times=1,perc=0,word_lib_path=r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_rmlowfreq_1221.csv'):
    data = du.get_data(file_route)
    doc_list = data.split('\n')[:-1]
    doc_num = len(doc_list)
    word_lib_df = du.get_word_lib_df(word_lib_path)
    word_lib_num = word_lib_df.shape[0]
    label_list = []
    tfc = []
    col = []
    row = []
    doc_no = 0
    for doc in tqdm(doc_list):
        label_list.append(doc.split('\t')[0])
        doc_word_list = doc.split('\t')[2].strip().split(' ')
        title = doc.split('\t')[1]
        doc_word_counter = Counter(doc_word_list)
        doc_lenth = len(doc_word_list)
        tmp_tf_idf_list=[]
        abst_words=set(doc_word_list[:int(doc_lenth*perc)])
        for word, count in doc_word_counter.items():
            try:
                if word in title:
                    count = count*title_times
                if perc!=0 and word in abst_words:
                    count = count*abst_times
                one_word_tf = count/doc_lenth
                one_word_idf = word_lib_df.loc[word, 'idf']
                one_word_tf_idf = math.log(one_word_tf+1,2)*one_word_idf #用log(tf+1)取代tf
                row.append(doc_no)
                col.append(word_lib_df.loc[word, 'index'])
                tmp_tf_idf_list.append(one_word_tf_idf)
            except:
                continue
        fenzi = math.sqrt(sum([x**2 for x in tmp_tf_idf_list]))
        tmp_tf_idf = [x/fenzi for x in tmp_tf_idf_list] #化为单位向量
        tfc.extend(tmp_tf_idf)
        doc_no += 1    
    tfc_train_feature_sparse_matrix = csr_matrix((tfc, (row, col)), shape=(doc_num, word_lib_num))
    le = LabelEncoder()
    le.fit(label_list)
    train_label_list = list(le.transform(label_list))
    return tfc_train_feature_sparse_matrix, train_label_list

def tf_idf_freqtimes(file_route,times): # 先把词频数乘个倍数，再求tf，所以一个文档里所有tf相加还是1
    data = du.get_data(file_route)
    doc_list = data.split('\n')[:-1]
    doc_num = len(doc_list)
    word_lib_df = du.get_word_lib_df()
    word_lib_num = word_lib_df.shape[0]
    label_list = []
    tf_idf = []
    col = []
    row = []
    doc_no = 0
    for doc in tqdm(doc_list):
        label_list.append(doc.split('\t')[0])
        title = doc.split('\t')[1]
        doc_word_list = doc.split('\t')[2].strip().split(' ')
        title_word = [x for x in doc_word_list if x in title ]
        for word in title_word:
            doc_word_list.extend([word]*(times-1))
        doc_word_counter = Counter(doc_word_list)
        doc_lenth = len(doc_word_list)
        for word, count in doc_word_counter.items():
            try:
                one_word_tf = count/doc_lenth
                one_word_idf = word_lib_df.loc[word, 'idf']  
                one_word_tf_idf = one_word_tf*one_word_idf
                row.append(doc_no)
                col.append(word_lib_df.loc[word, 'index'])
                tf_idf.append(one_word_tf_idf)
            except:
                continue
        doc_no += 1
    tf_idf_train_feature_sparse_matrix = csr_matrix((tf_idf, (row, col)), shape=(doc_num, word_lib_num))
    le = LabelEncoder()
    le.fit(label_list)
    train_label_list = list(le.transform(label_list))
    return tf_idf_train_feature_sparse_matrix, train_label_list

# 先把词频数乘个倍数，再求tf，所以一个文档里所有tf相加还是1
def tf_idf_tftimes(file_route,title_times=1,abst_times=1,perc=0):
    data = du.get_data(file_route)
    doc_list = data.split('\n')[:-1]
    doc_num = len(doc_list)
    word_lib_df = du.get_word_lib_df()
    word_lib_num = word_lib_df.shape[0]
    label_list = []
    tf_idf = []
    col = []
    row = []
    doc_no = 0
    for doc in tqdm(doc_list):
        label_list.append(doc.split('\t')[0])
        title = doc.split('\t')[1]
        doc_word_list = doc.split('\t')[2].strip().split(' ')
        doc_word_counter = Counter(doc_word_list)
        doc_lenth = len(doc_word_list)
        abst_words=set(doc_word_list[:int(doc_lenth*perc)])
        for word, count in doc_word_counter.items():
            try:
                if word in title:
                    count = count*title_times
                if perc!=0 and word in abst_words:
                    count = count*abst_times
                one_word_tf = count/doc_lenth
                one_word_idf = word_lib_df.loc[word, 'idf']  # 对于测试集有可能出现异常，即测试集中的某个词在词库中查不到因为词库是训练集构造的
                # fixme 用训练集构造的词库中报错说没有训练集第4543个doc中的nan这个词而实际上词库中有这个词 KeyError: 'the label [nan] is not in the [index]'
                one_word_tf_idf = one_word_tf*one_word_idf
                row.append(doc_no)
                col.append(word_lib_df.loc[word, 'index'])
                tf_idf.append(one_word_tf_idf)
            except:
                continue
        doc_no += 1
    tf_idf_train_feature_sparse_matrix = csr_matrix((tf_idf, (row, col)), shape=(doc_num, word_lib_num))
    # print(len(train_label_list), train_label_list[0])
    le = LabelEncoder()
    le.fit(label_list)
    # print(le.classes_)  # 显示出共八种标签['acq' 'crude' 'earn' 'grain' 'interest' 'money-fx' 'ship' 'trade']
    train_label_list = list(le.transform(label_list))
    # print(train_label_list[0: 5])  # 显示训练集中前五个文档的被编码后的标签[2, 0, 2, 2, 2]
    return tf_idf_train_feature_sparse_matrix, train_label_list

def tf_bdc_title(data0,title_times=1,abst_times=1,perc=0):
    word_lib_df = du.get_word_lib_df()
    label_list = [ '保险', '银行债券', '基金', '外汇', '期货', '科技', '股票','黄金']
    doc_label_list = []
    row = []
    col = []
    value = []
    doc_no = 0
    for i in tqdm(range(data0.shape[0])):
        title = data0.loc[i,'title']
        cate = data0.loc[i,'category_new']
        temp1 = word_lib_df.loc[data0.loc[i,'seg_list_rmstop_rmnum']]
        intitle_words = [x for x in set(data0.loc[i,'seg_list_rmstop_rmnum']) if x in title]
        temp1 = temp1[temp1['index']==temp1['index']]  ## 去除测试集中新的词，新的词index是nan，nan!=nan
        doc_label_list.append(data0.loc[i,'category_new'])        
        tmp_cnt = temp1.index.value_counts()
#        for word in tmp_cnt.index:
#            if word in title:
#                tmp_cnt[word] = tmp_cnt[word]*title_times
#        abst_words = set(data0.loc[i,'seg_list_rmstop_rmnum'][:int(perc*temp1.shape[0])])
        temp1 = pd.concat([temp1.drop_duplicates(),tmp_cnt],axis=1)       
        temp1['one_word_tf'] = temp1['word']/temp1['word'].sum()
#        if abst_times!=1 or title_times!=1:
#            for word in temp1.index:
#                if word in title:
#                    temp1.loc[word,'one_word_tf'] = temp1.loc[word,'one_word_tf']*title_times
#                if word in abst_words:
#                    temp1.loc[word,'one_word_tf'] = temp1.loc[word,'one_word_tf']*abst_times
        temp1['word_dc'] = [0]*temp1.shape[0]
        temp1['word_dc_pec'] = [0]*temp1.shape[0]
        for label in label_list:
    #        temp1[label+'_pec'] = [0]*temp1.shape[0]
            temp1[label+'_pec'] = temp1[label]/cate_count[label]
            if label==cate:
                try:
                    temp1.loc[intitle_words,label+'_pec'] = temp1.loc[intitle_words,label+'_pec']*3
                except:
                    continue
            temp1['word_dc_pec'] = temp1['word_dc_pec'] + temp1[label+'_pec']
        for label in label_list:
#            temp1[label+'_pec'] = temp1[label+'_pec']/temp1[label+'_pec'].sum()
            temp1[label+'_pec1'] = temp1[label+'_pec'] / temp1['word_dc_pec']
            temp1[label+'_dc'] = temp1[label+'_pec1'].apply(default0)
            temp1['word_dc'] = temp1['word_dc'] + temp1[label+'_dc']
        temp1['word_dc_final'] = [1]*temp1.shape[0] + temp1['word_dc']/math.log(len(label_list), 2)
    #        temp1[temp1[label]!=0][label+'_pec'] = temp1[temp1[label]!=0][label]/temp1[temp1[label]!=0]['doc_num']
    #        temp1[temp1[label]!=0][label+'_dc'] = temp1[temp1[label]!=0][label+'_pec'].apply(default0)
        temp1['tf_bdc'] = temp1['word_dc_final']*temp1['one_word_tf']
#        temp1['idf_bdc'] = temp1['word_dc_final']*temp1['idf']
        col_list = temp1['index'].tolist()
        value_list = temp1['tf_bdc'].tolist()
    #    doc_no = doc_no+temp1.shape[0]
        row_list = [doc_no]*temp1.shape[0]
        row.extend(row_list)
        col.extend(col_list)
        value.extend(value_list)
        doc_no += 1
        
    tf_dc_feature_sparse_matrix = csr_matrix((value, (row, col)), shape=(len(doc_label_list), word_lib_df.shape[0]))
    le = LabelEncoder()
    le.fit(doc_label_list)
    return tf_dc_feature_sparse_matrix, le.transform(doc_label_list)

def bdc_title(data0,title_times=1,abst_times=1,perc=0,word_lib_path=r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_rmlowfreq_1221.csv'):
    word_lib_df = du.get_word_lib_df(word_lib_path)
    label_list = [ '保险', '银行债券', '基金', '外汇', '期货', '科技', '股票','黄金']
    doc_label_list = []
    row = []
    col = []
    value = []
    doc_no = 0
    for i in tqdm(range(data0.shape[0])):
#        title = data0.loc[i,'title']
        temp1 = word_lib_df.loc[data0.loc[i,'seg_list_rmstop_rmnum']]
#        if title_times!=1:
#            intitle_words = [x for x in set(data0.loc[i,'seg_list_rmstop_rmnum']) if x in title]
#        if abst_times!=1:
#            abstract_words = data0.loc[i,'seg_list_rmstop_rmnum'][:int(len(data0.loc[i,'seg_list_rmstop_rmnum'])*0.1)]
        temp1 = temp1[temp1['index']==temp1['index']]  ## 去除测试集中新的词，新的词index是nan，nan!=nan
        doc_label_list.append(data0.loc[i,'category_new'])        
        tmp_cnt = temp1.index.value_counts()
        temp1 = pd.concat([temp1.drop_duplicates(),tmp_cnt],axis=1)       
        temp1['one_word_tf'] = temp1['word']/temp1['word'].sum()
        temp1['word_dc'] = [0]*temp1.shape[0]
        temp1['word_dc_pec'] = [0]*temp1.shape[0]
        for label in label_list:
            temp1[label+'_pec'] = temp1[label]/cate_count[label]
            temp1['word_dc_pec'] = temp1['word_dc_pec'] + temp1[label+'_pec']
        for label in label_list:
            temp1[label+'_pec1'] = temp1[label+'_pec'] / temp1['word_dc_pec']
            temp1[label+'_dc'] = temp1[label+'_pec1'].apply(default0)
            temp1['word_dc'] = temp1['word_dc'] + temp1[label+'_dc']
        temp1['word_dc_final'] = [1]*temp1.shape[0] + temp1['word_dc']/math.log(len(label_list), 2)
        col_list = temp1['index'].tolist()
        value_list = temp1['word_dc_final'].tolist()
        row_list = [doc_no]*temp1.shape[0]
        row.extend(row_list)
        col.extend(col_list)
        value.extend(value_list)
        doc_no += 1        
    feature_sparse_matrix = csr_matrix((value, (row, col)), shape=(len(doc_label_list), word_lib_df.shape[0]))
    le = LabelEncoder()
    le.fit(doc_label_list)
    return feature_sparse_matrix, le.transform(doc_label_list)

def dc(data0,title_times=1,abst_times=1,perc=0,word_lib_path=r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_rmlowfreq_1221.csv'):
    word_lib_df = du.get_word_lib_df(word_lib_path)
    label_list = [ '保险', '基金', '外汇', '期货', '科技', '股票','银行债券','黄金']
    doc_label_list = []
    row = []
    col = []
    value = []
    doc_no = 0
    for i in tqdm(range(data0.shape[0])):
        title = data0.loc[i,'title']
        temp1 = word_lib_df.loc[data0.loc[i,'seg_list_rmstop_rmnum']]
        temp1 = temp1[temp1['index']==temp1['index']]  ## 去除测试集中新的词，新的词index是nan，nan!=nan
        doc_label_list.append(data0.loc[i,'category_new'])        
        tmp_cnt = temp1.index.value_counts() 
        abst_words = set(data0.loc[i,'seg_list_rmstop_rmnum'][:int(perc*temp1.shape[0])])
        temp1 = pd.concat([temp1.drop_duplicates(),tmp_cnt],axis=1)
        temp1['one_word_tf'] = temp1['word']/temp1['word'].sum()
        if abst_times!=1 or title_times!=1:
            for word in temp1.index:
                if word in title:
                    temp1.loc[word,'one_word_tf'] = temp1.loc[word,'one_word_tf']*abst_times
                if word in abst_words:
                    temp1.loc[word,'one_word_tf'] = temp1.loc[word,'one_word_tf']*title_times
        temp1['word_dc'] = [0]*temp1.shape[0]
        for label in label_list:
    #        temp1[label+'_pec'] = [0]*temp1.shape[0]
            temp1[label+'_pec'] = temp1[label]/temp1['doc_num']
            temp1[label+'_dc'] = temp1[label+'_pec'].apply(default0)
            temp1['word_dc'] = temp1['word_dc'] + temp1[label+'_dc']
        temp1['word_dc_final'] = [1]*temp1.shape[0] + temp1['word_dc']/math.log(len(label_list), 2)
    #        temp1[temp1[label]!=0][label+'_pec'] = temp1[temp1[label]!=0][label]/temp1[temp1[label]!=0]['doc_num']
    #        temp1[temp1[label]!=0][label+'_dc'] = temp1[temp1[label]!=0][label+'_pec'].apply(default0)
#        temp1['tf_dc'] = temp1['word_dc_final']*temp1['one_word_tf']
        col_list = temp1['index'].tolist()
        value_list = temp1['word_dc_final'].tolist()
    #    doc_no = doc_no+temp1.shape[0]
        row_list = [doc_no]*temp1.shape[0]
        row.extend(row_list)
        col.extend(col_list)
        value.extend(value_list)
        doc_no += 1
        
    tf_dc_feature_sparse_matrix = csr_matrix((value, (row, col)), shape=(len(doc_label_list), word_lib_df.shape[0]))
    le = LabelEncoder()
    le.fit(doc_label_list)
    return tf_dc_feature_sparse_matrix, le.transform(doc_label_list)