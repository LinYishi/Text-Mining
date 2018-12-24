# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 10:40:24 2018

@author: DELL
"""

# 测试新样本
import pandas as pd
import re
import jieba
new_data_test=pd.read_csv(r'C:\Users\DELL\Desktop\article_hexun_1812.csv')
new_data_test.fillna('')
#def rm_claim(text):
#    return re.sub('本文转自.*?，财经369.*?责任','',text)
#new_data_test['body_text_rmc']=new_data_test['body_text'].apply(rm_claim)
def seg(text):
    try:
        seg_list=jieba.lcut(text)
    except AttributeError:
        seg_list=''
    return seg_list
new_data_test['seg_list'] = new_data_test['body_text'].apply(seg)
new_data_test=new_data_test[new_data_test['seg_list']!='']
stopwords=open(r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\stopwords1.txt','r',encoding='utf-8').readlines()
stopwords=[word.encode('utf-8').decode('utf-8-sig').strip() for word in stopwords]
stopwords_set=set(stopwords)

def rm_stpword(l1):
    return [x for x in l1 if x not in stopwords_set and x!='\n']

new_data_test['seg_list_rmstop'] = new_data_test['seg_list'].apply(rm_stpword)

new_data_test['category']=new_data_test['category'].replace('港股','股票')
invalid_item = new_data_test[new_data_test['title'].isna()]['author'].values
new_data_test = new_data_test[-new_data_test['author'].isin(invalid_item)]

new_data_test = new_data_test[new_data_test['body_text']!='']

def rm_space(lst):
    return [x.strip() for x in lst if ' ' not in x and '\n' not in x and '\t' not in x and x!='']

new_data_test['seg_list_rmstop'] = new_data_test['seg_list_rmstop'].apply(rm_space)

# 合并银行债券，去掉互金
def bind_cate(x):
    if x in ['银行','债券']:
        b= '银行债券'
    else:
        b= x
    return b
new_data_test['category_new'] = new_data_test['category'].apply(bind_cate)
new_data_test = new_data_test[new_data_test['category_new']!='互联网金融']

with open(r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\test_new_doc.txt','w',encoding='utf-8') as f:
    for i in range(new_data_test.shape[0]):
        categ = new_data_test.iloc[i,-1]
        title = new_data_test.iloc[i,1]
        word_list = new_data_test.iloc[i,-2]
        f.write(categ+'\t'+title+'\t'+' '.join(word_list)+'\n')
from sklearn.metrics import f1_score
#import pickle
import algorithm as algo
import classifier as clf
#import numpy as np
train_file_route = r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\train_new.txt'
test_file_route = r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\test_new_doc.txt'
def evaluate(test_label_list, predict_test_label_list):
    '''
    用于评估分类效果（决定分类效果的因素：词的权重算法的选择及分类器的选择和调参）
    :param test_label_list:
    :param predict_test_label_list:
    :return:
    '''
    Macro_F1 = f1_score(test_label_list, predict_test_label_list, average='macro')
    Micro_F1 = f1_score(test_label_list, predict_test_label_list, average='micro')
    print('Macro_F1:', Macro_F1)
    print('Micro_F1:', Micro_F1)
    return Macro_F1,Micro_F1
best_result_ontest=pd.DataFrame(columns=['algo','times','macro','micro'])
# 原始tf-idf在训练集上表现
train_feature_sparse_matrix, train_label_list = algo.tf_idf_tftimes(train_file_route)
test_feature_sparse_matrix, test_label_list = algo.tf_idf_tftimes(test_file_route)
predict_test_label_list = clf.SVM(train_feature_sparse_matrix, train_label_list, test_feature_sparse_matrix)
Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
# tf-idf in title 1.8
train_feature_sparse_matrix, train_label_list = algo.tf_idf_tftimes(train_file_route,title_times=1.8)
test_feature_sparse_matrix, test_label_list = algo.tf_idf_tftimes(test_file_route,title_times=1.8)
predict_test_label_list_1 = clf.SVM(train_feature_sparse_matrix, train_label_list, test_feature_sparse_matrix)
Macro_F1_1,Micro_F1_1=evaluate(test_label_list, predict_test_label_list_1)
# tf_idf in abst 0.1
train_feature_sparse_matrix, train_label_list = algo.tf_idf_tftimes(train_file_route,abst_times=1.6,perc=0.1)
test_feature_sparse_matrix, test_label_list = algo.tf_idf_tftimes(test_file_route,abst_times=1.6,perc=0.1)
predict_test_label_list_1 = clf.SVM(train_feature_sparse_matrix, train_label_list, test_feature_sparse_matrix)
Macro_F1_1,Micro_F1_1=evaluate(test_label_list, predict_test_label_list_1)

tmp=pd.DataFrame(['tf_idf',1,Macro_F1,Micro_F1]).T
tmp.columns=best_result_ontest.columns
best_result_ontest = best_result_ontest.append(tmp)

tmp=pd.DataFrame(['tf_idf_intitle',1.8,0.8519903445659065,0.9105675146771037]).T
tmp.columns=best_result_ontest.columns
best_result_ontest = best_result_ontest.append(tmp)

tmp=pd.DataFrame(['tf_idf_inabst',1.6,Macro_F1_1,Micro_F1_1]).T
tmp.columns=best_result_ontest.columns
best_result_ontest = best_result_ontest.append(tmp)

# tfc
train_feature_sparse_matrix, train_label_list = algo.tfc(train_file_route)
test_feature_sparse_matrix, test_label_list = algo.tfc(test_file_route)
predict_test_label_list = clf.SVM(train_feature_sparse_matrix, train_label_list, test_feature_sparse_matrix)
Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
tmp=pd.DataFrame(['tfc',1,Macro_F1,Micro_F1]).T
tmp.columns=best_result_ontest.columns
best_result_ontest = best_result_ontest.append(tmp)
# tfc in title 1.2
train_feature_sparse_matrix, train_label_list = algo.tfc(train_file_route,title_times=1.2)
test_feature_sparse_matrix, test_label_list = algo.tfc(test_file_route,title_times=1.2)
predict_test_label_list = clf.SVM(train_feature_sparse_matrix, train_label_list, test_feature_sparse_matrix)
Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
tmp=pd.DataFrame(['tfc_intitle',1.2,Macro_F1_1,Micro_F1]).T
tmp.columns=best_result_ontest.columns
best_result_ontest = best_result_ontest.append(tmp)
# tfc in abst 1.2
train_feature_sparse_matrix, train_label_list = algo.tfc(train_file_route,abst_times=1.2,perc=0.1)
test_feature_sparse_matrix, test_label_list = algo.tfc(test_file_route,abst_times=1.2,perc=0.1)
predict_test_label_list = clf.SVM(train_feature_sparse_matrix, train_label_list, test_feature_sparse_matrix)
Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
tmp=pd.DataFrame(['tfc_inabst',1.2,Macro_F1,Micro_F1]).T
tmp.columns=best_result_ontest.columns
best_result_ontest = best_result_ontest.append(tmp)

# itc
train_feature_sparse_matrix, train_label_list = algo.itc(train_file_route)
test_feature_sparse_matrix, test_label_list = algo.itc(test_file_route)
predict_test_label_list = clf.SVM(train_feature_sparse_matrix, train_label_list, test_feature_sparse_matrix)
Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
tmp=pd.DataFrame(['itc',1,Macro_F1,Micro_F1]).T
tmp.columns=best_result_ontest.columns
best_result_ontest = best_result_ontest.append(tmp)
# itc in title 1.4
train_feature_sparse_matrix, train_label_list = algo.itc(train_file_route,title_times=1.4)
test_feature_sparse_matrix, test_label_list = algo.itc(test_file_route,title_times=1.4)
predict_test_label_list = clf.SVM(train_feature_sparse_matrix, train_label_list, test_feature_sparse_matrix)
Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
tmp=pd.DataFrame(['itc_intitle',1.4,Macro_F1,Micro_F1]).T
tmp.columns=best_result_ontest.columns
best_result_ontest = best_result_ontest.append(tmp)
# itc in abst 1.2
train_feature_sparse_matrix, train_label_list = algo.itc(train_file_route,abst_times=1.2,perc=0.1)
test_feature_sparse_matrix, test_label_list = algo.itc(test_file_route,abst_times=1.2,perc=0.1)
predict_test_label_list = clf.SVM(train_feature_sparse_matrix, train_label_list, test_feature_sparse_matrix)
Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
tmp=pd.DataFrame(['itc_inabst',1.2,Macro_F1,Micro_F1]).T
tmp.columns=best_result_ontest.columns
best_result_ontest = best_result_ontest.append(tmp)
import numpy as np
new_data_test.index=np.arange(new_data_test.shape[0])
# dc
import pickle
f=open(r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\train_rmlowfreq.pkl','rb')
data_train=pickle.load(f)
f.close()
new_data_test['seg_list_rmstop_rmnum']=new_data_test['seg_list_rmstop']
train_feature_sparse_matrix, train_label_list = algo.dc(data_train)
test_feature_sparse_matrix, test_label_list = algo.dc(new_data_test)
predict_test_label_list = clf.SVM(train_feature_sparse_matrix, train_label_list, test_feature_sparse_matrix)
Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
tmp=pd.DataFrame(['dc',1,Macro_F1,Micro_F1]).T
tmp.columns=best_result_ontest.columns
best_result_ontest = best_result_ontest.append(tmp)
# dc in title 5
train_feature_sparse_matrix, train_label_list = algo.dc(data_train,word_lib_path=r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_df_bdctitle5.csv')
test_feature_sparse_matrix, test_label_list = algo.dc(new_data_test,word_lib_path=r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_df_bdctitle5.csv')
predict_test_label_list = clf.SVM(train_feature_sparse_matrix, train_label_list, test_feature_sparse_matrix)
Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
tmp=pd.DataFrame(['dc_intitle',5,Macro_F1,Micro_F1]).T
tmp.columns=best_result_ontest.columns
best_result_ontest = best_result_ontest.append(tmp)
# dc in abst 2
train_feature_sparse_matrix, train_label_list = algo.dc(data_train,word_lib_path=r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_df_bdcabst2.csv')
test_feature_sparse_matrix, test_label_list = algo.dc(new_data_test,word_lib_path=r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_df_bdcabst2.csv')
predict_test_label_list = clf.SVM(train_feature_sparse_matrix, train_label_list, test_feature_sparse_matrix)
Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
tmp=pd.DataFrame(['dc_inabst',2,Macro_F1,Micro_F1]).T
tmp.columns=best_result_ontest.columns
best_result_ontest = best_result_ontest.append(tmp)

#bdc
train_feature_sparse_matrix, train_label_list = algo.bdc_title(data_train)
test_feature_sparse_matrix, test_label_list = algo.bdc_title(new_data_test)
predict_test_label_list = clf.SVM(train_feature_sparse_matrix, train_label_list, test_feature_sparse_matrix)
Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
tmp=pd.DataFrame(['bdc',1,Macro_F1,Micro_F1]).T
tmp.columns=best_result_ontest.columns
best_result_ontest = best_result_ontest.append(tmp)
# bdc in title 2
train_feature_sparse_matrix, train_label_list = algo.bdc_title(data_train,word_lib_path=r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_df_bdctitle2.csv')
test_feature_sparse_matrix, test_label_list = algo.bdc_title(new_data_test,word_lib_path=r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_df_bdctitle2.csv')
predict_test_label_list = clf.SVM(train_feature_sparse_matrix, train_label_list, test_feature_sparse_matrix)
Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
tmp=pd.DataFrame(['bdc_intitle',2,Macro_F1,Micro_F1]).T
tmp.columns=best_result_ontest.columns
best_result_ontest = best_result_ontest.append(tmp)
# bdc in abst 2
train_feature_sparse_matrix, train_label_list = algo.bdc_title(data_train,word_lib_path=r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_df_bdcabst2.csv')
test_feature_sparse_matrix, test_label_list = algo.bdc_title(new_data_test,word_lib_path=r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_df_bdcabst2.csv')
predict_test_label_list = clf.SVM(train_feature_sparse_matrix, train_label_list, test_feature_sparse_matrix)
Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
tmp=pd.DataFrame(['bdc_inabst',2,Macro_F1,Micro_F1]).T
tmp.columns=best_result_ontest.columns
best_result_ontest = best_result_ontest.append(tmp)

#tfdc
train_feature_sparse_matrix, train_label_list = algo.tf_dc(data_train)
test_feature_sparse_matrix, test_label_list = algo.tf_dc(new_data_test)
predict_test_label_list = clf.SVM(train_feature_sparse_matrix, train_label_list, test_feature_sparse_matrix)
Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
tmp=pd.DataFrame(['tfdc',1,Macro_F1,Micro_F1]).T
tmp.columns=best_result_ontest.columns
best_result_ontest = best_result_ontest.append(tmp)
# tfdc in title 10
train_feature_sparse_matrix, train_label_list = algo.tf_dc(data_train,word_lib_path=r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_df_bdctitle10.csv')
test_feature_sparse_matrix, test_label_list = algo.tf_dc(new_data_test,word_lib_path=r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_df_bdctitle10.csv')
predict_test_label_list = clf.SVM(train_feature_sparse_matrix, train_label_list, test_feature_sparse_matrix)
Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
tmp=pd.DataFrame(['tfdc_intitle',10,Macro_F1,Micro_F1]).T
tmp.columns=best_result_ontest.columns
best_result_ontest = best_result_ontest.append(tmp)
# tfdc in abst 10
train_feature_sparse_matrix, train_label_list = algo.tf_dc(data_train,word_lib_path=r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_df_bdcabst10.csv')
test_feature_sparse_matrix, test_label_list = algo.tf_dc(new_data_test,word_lib_path=r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_df_bdcabst10.csv')
predict_test_label_list = clf.SVM(train_feature_sparse_matrix, train_label_list, test_feature_sparse_matrix)
Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
tmp=pd.DataFrame(['tfdc_inabst',10,Macro_F1,Micro_F1]).T
tmp.columns=best_result_ontest.columns
best_result_ontest = best_result_ontest.append(tmp)

#tfbdc
train_feature_sparse_matrix, train_label_list = algo.tf_bdc(data_train)
test_feature_sparse_matrix, test_label_list = algo.tf_bdc(new_data_test)
predict_test_label_list = clf.SVM(train_feature_sparse_matrix, train_label_list, test_feature_sparse_matrix)
Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
tmp=pd.DataFrame(['tfbdc',1,Macro_F1,Micro_F1]).T
tmp.columns=best_result_ontest.columns
best_result_ontest = best_result_ontest.append(tmp)
# tfbdc in title 10
train_feature_sparse_matrix, train_label_list = algo.tf_bdc(data_train,word_lib_path=r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_df_bdctitle10.csv')
test_feature_sparse_matrix, test_label_list = algo.tf_bdc(new_data_test,word_lib_path=r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_df_bdctitle10.csv')
predict_test_label_list = clf.SVM(train_feature_sparse_matrix, train_label_list, test_feature_sparse_matrix)
Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
tmp=pd.DataFrame(['tfbdc_intitle',10,Macro_F1,Micro_F1]).T
tmp.columns=best_result_ontest.columns
best_result_ontest = best_result_ontest.append(tmp)
# tfbdc in abst 10
train_feature_sparse_matrix, train_label_list = algo.tf_bdc(data_train,word_lib_path=r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_df_bdcabst10.csv')
test_feature_sparse_matrix, test_label_list = algo.tf_bdc(new_data_test,word_lib_path=r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_df_bdcabst10.csv')
predict_test_label_list = clf.SVM(train_feature_sparse_matrix, train_label_list, test_feature_sparse_matrix)
Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
tmp=pd.DataFrame(['tfbdc_inabst',10,Macro_F1,Micro_F1]).T
tmp.columns=best_result_ontest.columns
best_result_ontest = best_result_ontest.append(tmp)






######## 看看混淆矩阵

# tf-idf in abst 1.6
train_feature_sparse_matrix, train_label_list = algo.tf_idf_tftimes(train_file_route,abst_times=1.6,perc=0.1)
test_feature_sparse_matrix, test_label_list = algo.tf_idf_tftimes(test_file_route,abst_times=1.6,perc=0.1)
predict_test_label_list = clf.SVM(train_feature_sparse_matrix, train_label_list, test_feature_sparse_matrix)
Macro_F1_1,Micro_F1_1=evaluate(test_label_list, predict_test_label_list)
test_label_list =list(pd.Series( test_label_list).map({0:'保险', 1:'基金', 2:'外汇', 3:'期货', 4:'科技', 5:'股票',6:'银行债券',7:'黄金'}))
predict_test_label_list = list(pd.Series( predict_test_label_list).map({0:'保险', 1:'基金', 2:'外汇', 3:'期货', 4:'科技', 5:'股票',6:'银行债券',7:'黄金'}))

# confusion matrix
import matplotlib
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
zhfont1 = matplotlib.font_manager.FontProperties(fname=r'C:\Users\DELL\Desktop\SIMYOU.TTF')
classes_txt     = ['保险',  '基金', '外汇', '期货', '科技', '股票','银行债券','黄金']
def plot_confusion_matrix(cm,class_,title='Confusion Matrix',cmap=plt.cm.Reds):
    """
    This function plots a confusion matrix
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(class_))
    plt.xticks(tick_marks, class_, rotation=90, fontproperties=zhfont1)
    plt.yticks(tick_marks, class_,fontproperties=zhfont1)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('真实标签', fontproperties=zhfont1,rotation=90)
    plt.xlabel('预测标签', fontproperties=zhfont1)
    plt.show()

plot_confusion_matrix(confusion_matrix(test_label_list, predict_test_label_list),classes_txt,title='Confusion Matrix',cmap=plt.cm.Reds)





train_feature_sparse_matrix, train_label_list = algo.tf_idf_tftimes(train_file_route)
test_feature_sparse_matrix, test_label_list = algo.tf_idf_tftimes(test_file_route)
predict_test_label_list = clf.SVM(train_feature_sparse_matrix, train_label_list, test_feature_sparse_matrix)
Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
test_label_list =list(pd.Series( test_label_list).map({0:'保险', 1:'基金', 2:'外汇', 3:'期货', 4:'科技', 5:'股票',6:'银行债券',7:'黄金'}))
predict_test_label_list = list(pd.Series( predict_test_label_list).map({0:'保险', 1:'基金', 2:'外汇', 3:'期货', 4:'科技', 5:'股票',6:'银行债券',7:'黄金'}))
plot_confusion_matrix(confusion_matrix(test_label_list, predict_test_label_list),classes_txt,title='Confusion Matrix',cmap=plt.cm.Reds)







