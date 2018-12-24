from sklearn.metrics import f1_score
import pickle
import algorithm as algo
import classifier as clf
import numpy as np
import pandas as pd
def get_test_label_list(train_file_route, test_file_route, algorithm, classifier,title_times=1,abst_times=1,perc=0):
    if algorithm == 'tf_idf':
        train_feature_sparse_matrix, train_label_list = algo.tf_idf(train_file_route)
        test_feature_sparse_matrix, test_label_list = algo.tf_idf(test_file_route)
    elif algorithm == 'tf_dc':
        train_feature_sparse_matrix, train_label_list = algo.tf_dc(train_file_route,title_times,abst_times,perc)
        test_feature_sparse_matrix, test_label_list = algo.tf_dc(test_file_route,title_times,abst_times,perc)
    elif algorithm == 'tf_bdc':
        train_feature_sparse_matrix, train_label_list = algo.tf_bdc(train_file_route,title_times,abst_times,perc)
        test_feature_sparse_matrix, test_label_list = algo.tf_bdc(test_file_route,title_times,abst_times,perc)
    elif algorithm == 'tfc':
        train_feature_sparse_matrix, train_label_list = algo.tfc(train_file_route,title_times,abst_times,perc)
        test_feature_sparse_matrix, test_label_list = algo.tfc(test_file_route,title_times,abst_times,perc)
    elif algorithm == 'tf_idf_freqtimes':
        train_feature_sparse_matrix, train_label_list = algo.tf_idf_freqtimes(train_file_route,title_times)
        test_feature_sparse_matrix, test_label_list = algo.tf_idf_freqtimes(test_file_route,title_times)
    elif algorithm == 'tf_idf_tftimes':
        train_feature_sparse_matrix, train_label_list = algo.tf_idf_tftimes(train_file_route,title_times,abst_times,perc)
        test_feature_sparse_matrix, test_label_list = algo.tf_idf_tftimes(test_file_route,title_times,abst_times,perc)
    else :## itc
        train_feature_sparse_matrix, train_label_list = algo.itc(train_file_route,title_times,abst_times,perc)
        test_feature_sparse_matrix, test_label_list = algo.itc(test_file_route,title_times,abst_times,perc)
    if classifier == 'KNN':
        predict_test_label_list = clf.KNN(train_feature_sparse_matrix, train_label_list, test_feature_sparse_matrix)
    else:
        predict_test_label_list = clf.SVM(train_feature_sparse_matrix, train_label_list, test_feature_sparse_matrix)
    return test_label_list, predict_test_label_list


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

def get_pkl(file_route):
    f = open(file_route,'rb')
    data =pickle.load(f)
    f.close()
    return data

if __name__ == '__main__':
    train_file_route = r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\train_new.txt'
    test_file_route = r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\test_new.txt'
    ## based on tf-idf
    test_label_list, predict_test_label_list = get_test_label_list(train_file_route, test_file_route, 'tf_idf', 'SVM',times=1)
    evaluate(test_label_list, predict_test_label_list)
    ## based on entrophy
    train_pkl_route = r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\train_rmlowfreq.pkl'
    test_pkl_route = r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\test_rmlowfreq.pkl'
    data_train = get_pkl(train_pkl_route)
    data_test = get_pkl(test_pkl_route)
    test_label_list, predict_test_label_list = get_test_label_list(data_train, data_test, 'tf_bdc', 'SVM',times=1)
    evaluate(test_label_list, predict_test_label_list)

score_list=[]
for i in range(2,11):
    test_label_list, predict_test_label_list = get_test_label_list(train_file_route, test_file_route, 'tf_idf_tf5', 'SVM',i)
    score_list.append(evaluate(test_label_list, predict_test_label_list))
import pandas as pd
t=pd.DataFrame(score_list,columns=['macro','micro'])
index=[7,8]
t.index=index
['']


##################
##  循环测试未合并的
##################
import pandas as pd
import numpy as np
results=pd.DataFrame(columns=['algo','times','macro','micro'])

for i in np.arange(0.1,1,0.1):
    test_label_list, predict_test_label_list = get_test_label_list(train_file_route, test_file_route, 'tf_idf_tftimes', 'SVM',times=i)
    Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
    tmp=pd.DataFrame(['tf_idf',i,Macro_F1,Micro_F1]).T
    tmp.columns=results.columns
    results = results.append(tmp)
for i in np.arange(0.1,1,0.1):
    test_label_list, predict_test_label_list = get_test_label_list(train_file_route, test_file_route, 'itc', 'SVM',times=i)
    Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
    tmp=pd.DataFrame(['itc',i,Macro_F1,Micro_F1]).T
    tmp.columns=results.columns
    results = results.append(tmp)
for i in np.arange(0.1,1,0.1):
    test_label_list, predict_test_label_list = get_test_label_list(train_file_route, test_file_route, 'tfc', 'SVM',times=i)
    Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
    tmp=pd.DataFrame(['tfc',i,Macro_F1,Micro_F1]).T
    tmp.columns=results.columns
    results = results.append(tmp)
for i in np.arange(2.1,5.0,0.1):
    test_label_list, predict_test_label_list = get_test_label_list(train_file_route, test_file_route, 'tf_idf_tftimes', 'SVM',times=i)
    Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
    tmp=pd.DataFrame(['tf_idf',i,Macro_F1,Micro_F1]).T
    tmp.columns=results.columns
    results = results.append(tmp)
results.to_csv(r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\results.csv')

for i in range(4,11):
    test_label_list, predict_test_label_list = get_test_label_list(train_file_route, test_file_route, 'tf_idf_tftimes', 'SVM',times=i)
    Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
    tmp=pd.DataFrame(['tf_idf',i,Macro_F1,Micro_F1]).T
    tmp.columns=results.columns
    results = results.append(tmp)
    test_label_list, predict_test_label_list = get_test_label_list(train_file_route, test_file_route, 'tfc', 'SVM',times=i)
    Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
    tmp=pd.DataFrame(['tfc',i,Macro_F1,Micro_F1]).T
    tmp.columns=results.columns
    results = results.append(tmp)
    test_label_list, predict_test_label_list = get_test_label_list(train_file_route, test_file_route, 'itc', 'SVM',times=i)
    Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
    tmp=pd.DataFrame(['itc',i,Macro_F1,Micro_F1]).T
    tmp.columns=results.columns
    results = results.append(tmp)
    
for i in range(1,7):
    test_label_list, predict_test_label_list = get_test_label_list(data_train, data_test, 'tf_bdc', 'SVM',times=i)
    Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
    tmp=pd.DataFrame(['tf_bdc',i,Macro_F1,Micro_F1]).T
    tmp.columns=results.columns
    results = results.append(tmp)
    test_label_list, predict_test_label_list = get_test_label_list(data_train, data_test, 'tf_dc', 'SVM',times=i)
    Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
    tmp=pd.DataFrame(['tf_dc',i,Macro_F1,Micro_F1]).T
    tmp.columns=results.columns
    results = results.append(tmp)
for i in range(7,11):
    test_label_list, predict_test_label_list = get_test_label_list(data_train, data_test, 'tf_bdc', 'SVM',times=i)
    Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
    tmp=pd.DataFrame(['tf_bdc',i,Macro_F1,Micro_F1]).T
    tmp.columns=results.columns
    results = results.append(tmp)
    test_label_list, predict_test_label_list = get_test_label_list(data_train, data_test, 'tf_dc', 'SVM',times=i)
    Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
    tmp=pd.DataFrame(['tf_dc',i,Macro_F1,Micro_F1]).T
    tmp.columns=results.columns
    results = results.append(tmp)

##################
##  循环测试合并的
##################
import pandas as pd
import numpy as np
results_combined=pd.DataFrame(columns=['algo','times','macro','micro'])
for i in range(1,11):
    test_label_list, predict_test_label_list = get_test_label_list(train_file_route, test_file_route, 'tf_idf_tftimes', 'SVM',times=i)
    Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
    tmp=pd.DataFrame(['tf_idf',i,Macro_F1,Micro_F1]).T
    tmp.columns=results_combined.columns
    results_combined = results_combined.append(tmp)
    test_label_list, predict_test_label_list = get_test_label_list(train_file_route, test_file_route, 'tfc', 'SVM',times=i)
    Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
    tmp=pd.DataFrame(['tfc',i,Macro_F1,Micro_F1]).T
    tmp.columns=results_combined.columns
    results_combined = results_combined.append(tmp)
    test_label_list, predict_test_label_list = get_test_label_list(train_file_route, test_file_route, 'itc', 'SVM',times=i)
    Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
    tmp=pd.DataFrame(['itc',i,Macro_F1,Micro_F1]).T
    tmp.columns=results_combined.columns
    results_combined = results_combined.append(tmp)

results_combined.to_csv(r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\results_combined.csv')


## 出现在前百分之，加权重
# tf_idf
test_label_list, predict_test_label_list = get_test_label_list(train_file_route, test_file_route, 'tf_idf_tftimes', 'SVM',times=1.1,perc=0.1)
Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
# tfc
test_label_list, predict_test_label_list = get_test_label_list(train_file_route, test_file_route, 'tfc', 'SVM',times=4,perc=0.1)
Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
# itc
test_label_list, predict_test_label_list = get_test_label_list(train_file_route, test_file_route, 'itc', 'SVM',times=1.1,perc=0.1)
Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
# tf_dc
test_label_list, predict_test_label_list = get_test_label_list(data_train, data_test, 'tf_dc', 'SVM',times=2,perc=0.1)
Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
# tf_bdc
test_label_list, predict_test_label_list = get_test_label_list(data_train, data_test, 'tf_bdc', 'SVM',times=2,perc=0.1)
Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)




tfc_abst = pd.DataFrame(columns=['algo','times','macro','micro'])
for i in np.arange(1.1,1.6,0.1):
    print(i)
    test_label_list, predict_test_label_list = get_test_label_list(train_file_route, test_file_route, 'tfc', 'SVM',times=i,perc=0.1)
    Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
    tmp=pd.DataFrame(['tfc',i,Macro_F1,Micro_F1]).T
    tmp.columns=tfc_abst.columns
    tfc_abst = tfc_abst.append(tmp)



# 标题 + 摘要
train_feature_sparse_matrix, train_label_list = algo.bdc_title(data_train,title_times=1)
test_feature_sparse_matrix, test_label_list = algo.bdc_title(data_test,title_times=1)
train_feature_sparse_matrix.data=np.nan_to_num(train_feature_sparse_matrix.data)
test_feature_sparse_matrix.data=np.nan_to_num(test_feature_sparse_matrix.data)
predict_test_label_list = clf.SVM(train_feature_sparse_matrix, train_label_list, test_feature_sparse_matrix)
Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)


# 12/18 1:05 跑循环
results=pd.DataFrame(columns=['algo','times','macro','micro'])
# tf-idf intitle 2.1-5.0
# tf-idf in abstract 1.1-3.0
# bdc in title 1-10
# bdc in abstract 1-10
# tfc,itc in title 1.1-2.9 
for i in np.arange(2.1,5.0,0.1):
    test_label_list, predict_test_label_list = get_test_label_list(train_file_route, test_file_route, 'tf_idf_tftimes', 'SVM',title_times=i)
    Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
    tmp=pd.DataFrame(['tf_idf_intitle',i,Macro_F1,Micro_F1]).T
    tmp.columns=results.columns
    results = results.append(tmp)
for i in np.arange(1.1,3.0,0.1):
    test_label_list, predict_test_label_list = get_test_label_list(train_file_route, test_file_route, 'tf_idf_tftimes', 'SVM',abst_times=i,perc=0.1)
    Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
    tmp=pd.DataFrame(['tf_idf_inabst',i,Macro_F1,Micro_F1]).T
    tmp.columns=results.columns
    results = results.append(tmp)
for i in np.arange(1,11):
    train_feature_sparse_matrix, train_label_list = algo.bdc_title(data_train,title_times=i)
    test_feature_sparse_matrix, test_label_list = algo.bdc_title(data_test,title_times=i)
    train_feature_sparse_matrix.data=np.nan_to_num(train_feature_sparse_matrix.data)
    test_feature_sparse_matrix.data=np.nan_to_num(test_feature_sparse_matrix.data)
    predict_test_label_list = clf.SVM(train_feature_sparse_matrix, train_label_list, test_feature_sparse_matrix)
    Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
    tmp=pd.DataFrame(['bdc_title',i,Macro_F1,Micro_F1]).T
    tmp.columns=results.columns
    results = results.append(tmp)
    
    train_feature_sparse_matrix, train_label_list = algo.bdc_title(data_train,abst_times=i,perc=0.1)
    test_feature_sparse_matrix, test_label_list = algo.bdc_title(data_test,abst_times=i,perc=0.1)
    train_feature_sparse_matrix.data=np.nan_to_num(train_feature_sparse_matrix.data)
    test_feature_sparse_matrix.data=np.nan_to_num(test_feature_sparse_matrix.data)
    predict_test_label_list = clf.SVM(train_feature_sparse_matrix, train_label_list, test_feature_sparse_matrix)
    Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
    tmp=pd.DataFrame(['bdc_abstract',i,Macro_F1,Micro_F1]).T
    tmp.columns=results.columns
    results = results.append(tmp)

for i in np.arange(1.1,3.0,0.1):
    test_label_list, predict_test_label_list = get_test_label_list(train_file_route, test_file_route, 'tfc', 'SVM',title_times=i)
    Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
    tmp=pd.DataFrame(['tfc_intitle',i,Macro_F1,Micro_F1]).T
    tmp.columns=results.columns
    results = results.append(tmp)
    test_label_list, predict_test_label_list = get_test_label_list(train_file_route, test_file_route, 'itc', 'SVM',title_times=i)
    Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
    tmp=pd.DataFrame(['itc_intitle',i,Macro_F1,Micro_F1]).T
    tmp.columns=results.columns
    results = results.append(tmp)

results.to_csv(r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\results_20181218.csv')

train_feature_sparse_matrix, train_label_list = algo.bdc_title(data_train,abst_times=15,perc=0.1)
test_feature_sparse_matrix, test_label_list = algo.bdc_title(data_test,abst_times=15,perc=0.1)
train_feature_sparse_matrix.data=np.nan_to_num(train_feature_sparse_matrix.data)
test_feature_sparse_matrix.data=np.nan_to_num(test_feature_sparse_matrix.data)
predict_test_label_list = clf.SVM(train_feature_sparse_matrix, train_label_list, test_feature_sparse_matrix)
Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
tmp=pd.DataFrame(['bdc_abstract',15,Macro_F1,Micro_F1]).T
tmp.columns=results.columns
results = results.append(tmp)


test_label_list, predict_test_label_list = get_test_label_list(train_file_route, test_file_route, 'tf_idf_tftimes', 'SVM',times=1,perc=0.2)
Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)



### bdc in title correct version
import imp
imp.reload(algo)
# title words *5
train_feature_sparse_matrix, train_label_list = algo.bdc_title(data_train)
test_feature_sparse_matrix, test_label_list = algo.bdc_title(data_test)
train_feature_sparse_matrix.data=np.nan_to_num(train_feature_sparse_matrix.data)
test_feature_sparse_matrix.data=np.nan_to_num(test_feature_sparse_matrix.data)
predict_test_label_list = clf.SVM(train_feature_sparse_matrix, train_label_list, test_feature_sparse_matrix)
Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
# micro:0.8925472179683513 macro:0.8592614094149624

# title words *3
import imp
imp.reload(algo)
train_feature_sparse_matrix, train_label_list = algo.bdc_title(data_train)
test_feature_sparse_matrix, test_label_list = algo.bdc_title(data_test)
train_feature_sparse_matrix.data=np.nan_to_num(train_feature_sparse_matrix.data)
test_feature_sparse_matrix.data=np.nan_to_num(test_feature_sparse_matrix.data)
predict_test_label_list = clf.SVM(train_feature_sparse_matrix, train_label_list, test_feature_sparse_matrix)
Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
# micro:0.892802450229709 macro:0.8593685835621189

# title words *2
import imp
imp.reload(algo)
train_feature_sparse_matrix, train_label_list = algo.bdc_title(data_train)
test_feature_sparse_matrix, test_label_list = algo.bdc_title(data_test)
train_feature_sparse_matrix.data=np.nan_to_num(train_feature_sparse_matrix.data)
test_feature_sparse_matrix.data=np.nan_to_num(test_feature_sparse_matrix.data)
predict_test_label_list = clf.SVM(train_feature_sparse_matrix, train_label_list, test_feature_sparse_matrix)
Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
# micro:0.8925472179683513 macro:0.859071087910259


## title words *1 update data_util.get_word_lib_df, reload algo,then run following codes
import imp
imp.reload(algo)
train_feature_sparse_matrix, train_label_list = algo.bdc_title(data_train)
test_feature_sparse_matrix, test_label_list = algo.bdc_title(data_test)
train_feature_sparse_matrix.data=np.nan_to_num(train_feature_sparse_matrix.data)
test_feature_sparse_matrix.data=np.nan_to_num(test_feature_sparse_matrix.data)
predict_test_label_list = clf.SVM(train_feature_sparse_matrix, train_label_list, test_feature_sparse_matrix)
Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
# micro:0.8933129147524247 macro:0.860195599254589

## abst words *3 
import imp
imp.reload(algo)
train_feature_sparse_matrix, train_label_list = algo.bdc_title(data_train)
test_feature_sparse_matrix, test_label_list = algo.bdc_title(data_test)
train_feature_sparse_matrix.data=np.nan_to_num(train_feature_sparse_matrix.data)
test_feature_sparse_matrix.data=np.nan_to_num(test_feature_sparse_matrix.data)
predict_test_label_list = clf.SVM(train_feature_sparse_matrix, train_label_list, test_feature_sparse_matrix)
Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
# micro:0.8933129147524247 macro:0.859332531955942





####### 2018/12/21 itc,tfc靠前文本,btc原始，加标题/靠前文本 实验 
results=pd.DataFrame(columns=['algo','times','macro','micro'])
# itc,tfc 靠前文本
for i in np.arange(1,11,0.2):
    # tf-idf
    test_label_list, predict_test_label_list = get_test_label_list(train_file_route, test_file_route, 'tf_idf_tftimes', 'SVM',title_times=i)
    Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
    tmp=pd.DataFrame(['tfidf_intitle',i,Macro_F1,Micro_F1]).T
    tmp.columns=results.columns
    results = results.append(tmp)
    test_label_list, predict_test_label_list = get_test_label_list(train_file_route, test_file_route, 'tf_idf_tftimes', 'SVM',abst_times=i,perc=0.1)
    Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
    tmp=pd.DataFrame(['tfidf_inabst',i,Macro_F1,Micro_F1]).T
    tmp.columns=results.columns
    results = results.append(tmp)
    # tfc
    test_label_list, predict_test_label_list = get_test_label_list(train_file_route, test_file_route, 'tfc', 'SVM',title_times=i)
    Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
    tmp=pd.DataFrame(['tfc_intitle',i,Macro_F1,Micro_F1]).T
    tmp.columns=results.columns
    results = results.append(tmp)
    test_label_list, predict_test_label_list = get_test_label_list(train_file_route, test_file_route, 'tfc', 'SVM',abst_times=i,perc=0.1)
    Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
    tmp=pd.DataFrame(['tfc_inabst',i,Macro_F1,Micro_F1]).T
    tmp.columns=results.columns
    results = results.append(tmp)
    # itc
    test_label_list, predict_test_label_list = get_test_label_list(train_file_route, test_file_route, 'itc', 'SVM',title_times=i)
    Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
    tmp=pd.DataFrame(['itc_intitle',i,Macro_F1,Micro_F1]).T
    tmp.columns=results.columns
    results = results.append(tmp)
    test_label_list, predict_test_label_list = get_test_label_list(train_file_route, test_file_route, 'itc', 'SVM',abst_times=i,perc=0.1)
    Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
    tmp=pd.DataFrame(['itc_inabst',i,Macro_F1,Micro_F1]).T
    tmp.columns=results.columns
    results = results.append(tmp)


# bdc 加标题权重，靠前文本的word_lib
import data_util as du
import imp
imp.reload(du)
def get_word_lib_df(path):
    word_lib_df = pd.read_csv(path,encoding='utf-8',sep=',',engine='python')
    word_lib_df.set_index('word', inplace=True)
    return word_lib_df
word_lib_df = get_word_lib_df(r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_rmlowfreq_1221.csv')

def in_title(word_list,title):
    intitle_words=[]
    for word in set(word_list):
        if word in title:
            intitle_words.append(word)
    return set(intitle_words)
##word_lib_df_bdctitle2 = word_lib_df.copy()
data_train['intitle_words'] = data_train.apply(lambda y:in_title(y['seg_list_rmstop_rmnum'],y['title']),axis=1)
def in_abstract(word_list,perc=0.1):
    return set(word_list[:int(len(word_list)*perc)])
data_train['inabst_words']=data_train['seg_list_rmstop_rmnum'].apply(in_abstract)

#word_lib_df.shape
from tqdm import tqdm
intitle_pathlist=[]
inabst_pathlist=[]

word_lib_bdctitle2=word_lib_df.copy()
for i in tqdm(range(data_train.shape[0])):
    for word in data_train.loc[i,'intitle_words']:
        word_lib_bdctitle2.loc[word,data_train.loc[i,'category_new']]+=1
word_lib_bdctitle2_route = r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_df_bdctitle2.csv'
word_lib_bdctitle2.to_csv(word_lib_bdctitle2_route)
intitle_pathlist.append(word_lib_bdctitle2_route)
del word_lib_bdctitle2
word_lib_bdctitle3=word_lib_df.copy()
for i in tqdm(range(data_train.shape[0])):
    for word in data_train.loc[i,'intitle_words']:
        word_lib_bdctitle3.loc[word,data_train.loc[i,'category_new']]+=2
word_lib_bdctitle3_route = r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_df_bdctitle3.csv'
word_lib_bdctitle3.to_csv(word_lib_bdctitle3_route)
intitle_pathlist.append(word_lib_bdctitle3_route)
del word_lib_bdctitle3
word_lib_bdctitle4=word_lib_df.copy()
for i in tqdm(range(data_train.shape[0])):
    for word in data_train.loc[i,'intitle_words']:
        word_lib_bdctitle4.loc[word,data_train.loc[i,'category_new']]+=3
word_lib_bdctitle4_route = r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_df_bdctitle4.csv'
word_lib_bdctitle4.to_csv(word_lib_bdctitle4_route)
intitle_pathlist.append(word_lib_bdctitle4_route)
del word_lib_bdctitle4
word_lib_bdctitle5=word_lib_df.copy()
for i in tqdm(range(data_train.shape[0])):
    for word in data_train.loc[i,'intitle_words']:
        word_lib_bdctitle5.loc[word,data_train.loc[i,'category_new']]+=4
word_lib_bdctitle5_route = r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_df_bdctitle5.csv'
word_lib_bdctitle5.to_csv(word_lib_bdctitle5_route)
intitle_pathlist.append(word_lib_bdctitle5_route)
del word_lib_bdctitle5
word_lib_bdctitle6=word_lib_df.copy()
for i in tqdm(range(data_train.shape[0])):
    for word in data_train.loc[i,'intitle_words']:
        word_lib_bdctitle6.loc[word,data_train.loc[i,'category_new']]+=5
word_lib_bdctitle6_route = r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_df_bdctitle6.csv'
word_lib_bdctitle6.to_csv(word_lib_bdctitle6_route)
intitle_pathlist.append(word_lib_bdctitle6_route)
del word_lib_bdctitle6
word_lib_bdctitle7=word_lib_df.copy()
for i in tqdm(range(data_train.shape[0])):
    for word in data_train.loc[i,'intitle_words']:
        word_lib_bdctitle7.loc[word,data_train.loc[i,'category_new']]+=6
word_lib_bdctitle7_route = r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_df_bdctitle7.csv'
word_lib_bdctitle7.to_csv(word_lib_bdctitle7_route)
intitle_pathlist.append(word_lib_bdctitle7_route)
del word_lib_bdctitle7
word_lib_bdctitle8=word_lib_df.copy()
for i in tqdm(range(data_train.shape[0])):
    for word in data_train.loc[i,'intitle_words']:
        word_lib_bdctitle8.loc[word,data_train.loc[i,'category_new']]+=7
word_lib_bdctitle8_route = r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_df_bdctitle8.csv'
word_lib_bdctitle8.to_csv(word_lib_bdctitle8_route)
intitle_pathlist.append(word_lib_bdctitle8_route)
del word_lib_bdctitle8
word_lib_bdctitle9=word_lib_df.copy()
for i in tqdm(range(data_train.shape[0])):
    for word in data_train.loc[i,'intitle_words']:
        word_lib_bdctitle9.loc[word,data_train.loc[i,'category_new']]+=8
word_lib_bdctitle9_route = r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_df_bdctitle9.csv'
word_lib_bdctitle9.to_csv(word_lib_bdctitle9_route)
intitle_pathlist.append(word_lib_bdctitle9_route)
del word_lib_bdctitle9
word_lib_bdctitle10=word_lib_df.copy()
for i in tqdm(range(data_train.shape[0])):
    for word in data_train.loc[i,'intitle_words']:
        word_lib_bdctitle10.loc[word,data_train.loc[i,'category_new']]+=9
word_lib_bdctitle10_route = r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_df_bdctitle10.csv'
word_lib_bdctitle10.to_csv(word_lib_bdctitle10_route)
intitle_pathlist.append(word_lib_bdctitle10_route)
del word_lib_bdctitle10
## in abst bdc word_lib

word_lib_bdcabst2=word_lib_df.copy()
for i in tqdm(range(data_train.shape[0])):
    for word in data_train.loc[i,'inabst_words']:
        word_lib_bdcabst2.loc[word,data_train.loc[i,'category_new']]+=1
word_lib_bdcabst2_route = r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_df_bdcabst2.csv'
word_lib_bdcabst2.to_csv(word_lib_bdcabst2_route)
inabst_pathlist.append(word_lib_bdcabst2_route)
del word_lib_bdcabst2
word_lib_bdcabst3=word_lib_df.copy()
for i in tqdm(range(data_train.shape[0])):
    for word in data_train.loc[i,'inabst_words']:
        word_lib_bdcabst3.loc[word,data_train.loc[i,'category_new']]+=2
word_lib_bdcabst3_route = r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_df_bdcabst3.csv'
word_lib_bdcabst3.to_csv(word_lib_bdcabst3_route)
inabst_pathlist.append(word_lib_bdcabst3_route)
del word_lib_bdcabst3
word_lib_bdcabst4=word_lib_df.copy()
for i in tqdm(range(data_train.shape[0])):
    for word in data_train.loc[i,'inabst_words']:
        word_lib_bdcabst4.loc[word,data_train.loc[i,'category_new']]+=3
word_lib_bdcabst4_route = r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_df_bdcabst4.csv'
word_lib_bdcabst4.to_csv(word_lib_bdcabst4_route)
inabst_pathlist.append(word_lib_bdcabst4_route)
del word_lib_bdcabst4
word_lib_bdcabst5=word_lib_df.copy()
for i in tqdm(range(data_train.shape[0])):
    for word in data_train.loc[i,'inabst_words']:
        word_lib_bdcabst5.loc[word,data_train.loc[i,'category_new']]+=4
word_lib_bdcabst5_route = r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_df_bdcabst5.csv'
word_lib_bdcabst5.to_csv(word_lib_bdcabst5_route)
inabst_pathlist.append(word_lib_bdcabst5_route)
del word_lib_bdcabst5
word_lib_bdcabst6=word_lib_df.copy()
for i in tqdm(range(data_train.shape[0])):
    for word in data_train.loc[i,'inabst_words']:
        word_lib_bdcabst6.loc[word,data_train.loc[i,'category_new']]+=5
word_lib_bdcabst6_route = r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_df_bdcabst6.csv'
word_lib_bdcabst6.to_csv(word_lib_bdcabst6_route)
inabst_pathlist.append(word_lib_bdcabst6_route)
del word_lib_bdcabst6
word_lib_bdcabst7=word_lib_df.copy()
for i in tqdm(range(data_train.shape[0])):
    for word in data_train.loc[i,'inabst_words']:
        word_lib_bdcabst7.loc[word,data_train.loc[i,'category_new']]+=6
word_lib_bdcabst7_route = r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_df_bdcabst7.csv'
word_lib_bdcabst7.to_csv(word_lib_bdcabst7_route)
inabst_pathlist.append(word_lib_bdcabst7_route)
del word_lib_bdcabst7
word_lib_bdcabst8=word_lib_df.copy()
for i in tqdm(range(data_train.shape[0])):
    for word in data_train.loc[i,'inabst_words']:
        word_lib_bdcabst8.loc[word,data_train.loc[i,'category_new']]+=7
word_lib_bdcabst8_route = r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_df_bdcabst8.csv'
word_lib_bdcabst8.to_csv(word_lib_bdcabst8_route)
inabst_pathlist.append(word_lib_bdcabst8_route)
del word_lib_bdcabst8
word_lib_bdcabst9=word_lib_df.copy()
for i in tqdm(range(data_train.shape[0])):
    for word in data_train.loc[i,'inabst_words']:
        word_lib_bdcabst9.loc[word,data_train.loc[i,'category_new']]+=8
word_lib_bdcabst9_route = r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_df_bdcabst9.csv'
word_lib_bdcabst9.to_csv(word_lib_bdcabst9_route)
inabst_pathlist.append(word_lib_bdcabst9_route)
del word_lib_bdcabst9
word_lib_bdcabst10=word_lib_df.copy()
for i in tqdm(range(data_train.shape[0])):
    for word in data_train.loc[i,'inabst_words']:
        word_lib_bdcabst10.loc[word,data_train.loc[i,'category_new']]+=9
word_lib_bdcabst10_route = r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_df_bdcabst10.csv'
word_lib_bdcabst10.to_csv(word_lib_bdcabst10_route)
inabst_pathlist.append(word_lib_bdcabst10_route)
del word_lib_bdcabst10
imp.reload(algo)

results=pd.DataFrame(columns=['algo','times','macro','micro'])
for i in range(4,11):
    #bdc
    word_lib_path=intitle_pathlist[i-2]
    train_feature_sparse_matrix, train_label_list = algo.bdc_title(data_train,word_lib_path=word_lib_path)
    test_feature_sparse_matrix, test_label_list = algo.bdc_title(data_test,word_lib_path=word_lib_path)
    predict_test_label_list = clf.SVM(train_feature_sparse_matrix, train_label_list, test_feature_sparse_matrix)
    Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
    tmp=pd.DataFrame(['bdc_intitle',i,Macro_F1,Micro_F1]).T
    tmp.columns=results.columns
    results = results.append(tmp)
    word_lib_path=inabst_pathlist[i-2]
    train_feature_sparse_matrix, train_label_list = algo.bdc_title(data_train,word_lib_path=word_lib_path)
    test_feature_sparse_matrix, test_label_list = algo.bdc_title(data_test,word_lib_path=word_lib_path)
    predict_test_label_list = clf.SVM(train_feature_sparse_matrix, train_label_list, test_feature_sparse_matrix)
    Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
    tmp=pd.DataFrame(['bdc_inabst',i,Macro_F1,Micro_F1]).T
    tmp.columns=results.columns
    results = results.append(tmp)
    #dc
    word_lib_path=intitle_pathlist[i-2]
    train_feature_sparse_matrix, train_label_list = algo.dc(data_train)
    test_feature_sparse_matrix, test_label_list = algo.dc(data_test)
    predict_test_label_list = clf.SVM(train_feature_sparse_matrix, train_label_list, test_feature_sparse_matrix)
    Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
    tmp=pd.DataFrame(['dc_intitle',i,Macro_F1,Micro_F1]).T
    tmp.columns=results.columns
    results = results.append(tmp)
    word_lib_path=inabst_pathlist[i-2]
    train_feature_sparse_matrix, train_label_list = algo.dc(data_train,word_lib_path=word_lib_path)
    test_feature_sparse_matrix, test_label_list = algo.dc(data_test,word_lib_path=word_lib_path)
    predict_test_label_list = clf.SVM(train_feature_sparse_matrix, train_label_list, test_feature_sparse_matrix)
    Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
    tmp=pd.DataFrame(['dc_inabst',i,Macro_F1,Micro_F1]).T
    tmp.columns=results.columns
    results = results.append(tmp)
    #tf_bdc
    word_lib_path=intitle_pathlist[i-2]
    train_feature_sparse_matrix, train_label_list = algo.tf_bdc(data_train,word_lib_path=word_lib_path)
    test_feature_sparse_matrix, test_label_list = algo.tf_bdc(data_test,word_lib_path=word_lib_path)
    predict_test_label_list = clf.SVM(train_feature_sparse_matrix, train_label_list, test_feature_sparse_matrix)
    Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
    tmp=pd.DataFrame(['tf_bdc_intitle',i,Macro_F1,Micro_F1]).T
    tmp.columns=results.columns
    results = results.append(tmp)
    word_lib_path=inabst_pathlist[i-2]
    train_feature_sparse_matrix, train_label_list = algo.tf_bdc(data_train,word_lib_path=word_lib_path)
    test_feature_sparse_matrix, test_label_list = algo.tf_bdc(data_test,word_lib_path=word_lib_path)
    predict_test_label_list = clf.SVM(train_feature_sparse_matrix, train_label_list, test_feature_sparse_matrix)
    Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
    tmp=pd.DataFrame(['tf_bdc_inabst',i,Macro_F1,Micro_F1]).T
    tmp.columns=results.columns
    results = results.append(tmp)
    #tf_dc
    word_lib_path=intitle_pathlist[i-2]
    train_feature_sparse_matrix, train_label_list = algo.tf_dc(data_train,word_lib_path=word_lib_path)
    test_feature_sparse_matrix, test_label_list = algo.tf_dc(data_test,word_lib_path=word_lib_path)
    predict_test_label_list = clf.SVM(train_feature_sparse_matrix, train_label_list, test_feature_sparse_matrix)
    Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
    tmp=pd.DataFrame(['tf_dc_intitle',i,Macro_F1,Micro_F1]).T
    tmp.columns=results.columns
    results = results.append(tmp)
    word_lib_path=inabst_pathlist[i-2]
    train_feature_sparse_matrix, train_label_list = algo.tf_dc(data_train,word_lib_path=word_lib_path)
    test_feature_sparse_matrix, test_label_list = algo.tf_dc(data_test,word_lib_path=word_lib_path)
    predict_test_label_list = clf.SVM(train_feature_sparse_matrix, train_label_list, test_feature_sparse_matrix)
    Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
    tmp=pd.DataFrame(['tf_dc_inabst',i,Macro_F1,Micro_F1]).T
    tmp.columns=results.columns
    results = results.append(tmp)
    

word_lib_bdcabst11=word_lib_df.copy()
for i in tqdm(range(data_train.shape[0])):
    for word in data_train.loc[i,'inabst_words']:
        word_lib_bdcabst11.loc[word,data_train.loc[i,'category_new']]+=10
word_lib_bdcabst11_route = r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_df_bdcabst11.csv'
word_lib_bdcabst11.to_csv(word_lib_bdcabst11_route)
inabst_pathlist.append(word_lib_bdcabst11)
del word_lib_bdcabst11
word_lib_bdcabst12=word_lib_df.copy()
for i in tqdm(range(data_train.shape[0])):
    for word in data_train.loc[i,'inabst_words']:
        word_lib_bdcabst12.loc[word,data_train.loc[i,'category_new']]+=11
word_lib_bdcabst12_route = r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_df_bdcabst12.csv'
word_lib_bdcabst12.to_csv(word_lib_bdcabst12_route)
inabst_pathlist.append(word_lib_bdcabst12_route)
del word_lib_bdcabst12
word_lib_bdcabst13=word_lib_df.copy()
for i in tqdm(range(data_train.shape[0])):
    for word in data_train.loc[i,'inabst_words']:
        word_lib_bdcabst13.loc[word,data_train.loc[i,'category_new']]+=12
word_lib_bdcabst13_route = r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_df_bdcabst13.csv'
word_lib_bdcabst13.to_csv(word_lib_bdcabst13_route)
inabst_pathlist.append(word_lib_bdcabst13_route)
del word_lib_bdcabst13
word_lib_bdcabst14=word_lib_df.copy()
for i in tqdm(range(data_train.shape[0])):
    for word in data_train.loc[i,'inabst_words']:
        word_lib_bdcabst14.loc[word,data_train.loc[i,'category_new']]+=13
word_lib_bdcabst14_route = r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_df_bdcabst14.csv'
word_lib_bdcabst14.to_csv(word_lib_bdcabst14_route)
inabst_pathlist.append(word_lib_bdcabst14_route)
del word_lib_bdcabst14
word_lib_bdcabst15=word_lib_df.copy()
for i in tqdm(range(data_train.shape[0])):
    for word in data_train.loc[i,'inabst_words']:
        word_lib_bdcabst15.loc[word,data_train.loc[i,'category_new']]+=14
word_lib_bdcabst15_route = r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_df_bdcabst15.csv'
word_lib_bdcabst15.to_csv(word_lib_bdcabst15_route)
inabst_pathlist.append(word_lib_bdcabst15_route)
del word_lib_bdcabst15
word_lib_bdcabst16=word_lib_df.copy()
for i in tqdm(range(data_train.shape[0])):
    for word in data_train.loc[i,'inabst_words']:
        word_lib_bdcabst16.loc[word,data_train.loc[i,'category_new']]+=15
word_lib_bdcabst16_route = r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_df_bdcabst16.csv'
word_lib_bdcabst16.to_csv(word_lib_bdcabst16_route)
inabst_pathlist.append(word_lib_bdcabst16_route)
del word_lib_bdcabst16
word_lib_bdcabst17=word_lib_df.copy()
for i in tqdm(range(data_train.shape[0])):
    for word in data_train.loc[i,'inabst_words']:
        word_lib_bdcabst17.loc[word,data_train.loc[i,'category_new']]+=16
word_lib_bdcabst17_route = r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_df_bdcabst17.csv'
word_lib_bdcabst17.to_csv(word_lib_bdcabst17_route)
inabst_pathlist.append(word_lib_bdcabst17_route)
del word_lib_bdcabst17
word_lib_bdcabst18=word_lib_df.copy()
for i in tqdm(range(data_train.shape[0])):
    for word in data_train.loc[i,'inabst_words']:
        word_lib_bdcabst18.loc[word,data_train.loc[i,'category_new']]+=17
word_lib_bdcabst18_route = r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_df_bdcabst18.csv'
word_lib_bdcabst18.to_csv(word_lib_bdcabst18_route)
inabst_pathlist.append(word_lib_bdcabst18_route)
del word_lib_bdcabst18
word_lib_bdcabst19=word_lib_df.copy()
for i in tqdm(range(data_train.shape[0])):
    for word in data_train.loc[i,'inabst_words']:
        word_lib_bdcabst19.loc[word,data_train.loc[i,'category_new']]+=18
word_lib_bdcabst19_route = r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_df_bdcabst19.csv'
word_lib_bdcabst19.to_csv(word_lib_bdcabst19_route)
inabst_pathlist.append(word_lib_bdcabst19_route)
del word_lib_bdcabst19
word_lib_bdcabst20=word_lib_df.copy()
for i in tqdm(range(data_train.shape[0])):
    for word in data_train.loc[i,'inabst_words']:
        word_lib_bdcabst20.loc[word,data_train.loc[i,'category_new']]+=19
word_lib_bdcabst20_route = r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_df_bdcabst20.csv'
word_lib_bdcabst20.to_csv(word_lib_bdcabst20_route)
inabst_pathlist.append(word_lib_bdcabst20_route)
del word_lib_bdcabst20


word_lib_bdctitle11=word_lib_df.copy()
for i in tqdm(range(data_train.shape[0])):
    for word in data_train.loc[i,'intitle_words']:
        word_lib_bdctitle11.loc[word,data_train.loc[i,'category_new']]+=10
word_lib_bdctitle11_route = r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_df_bdctitle11.csv'
word_lib_bdctitle11.to_csv(word_lib_bdctitle11_route)
intitle_pathlist.append(word_lib_bdctitle11_route)
del word_lib_bdctitle11
word_lib_bdctitle12=word_lib_df.copy()
for i in tqdm(range(data_train.shape[0])):
    for word in data_train.loc[i,'intitle_words']:
        word_lib_bdctitle12.loc[word,data_train.loc[i,'category_new']]+=11
word_lib_bdctitle12_route = r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_df_bdctitle12.csv'
word_lib_bdctitle12.to_csv(word_lib_bdctitle12_route)
intitle_pathlist.append(word_lib_bdctitle12_route)
del word_lib_bdctitle12
word_lib_bdctitle13=word_lib_df.copy()
for i in tqdm(range(data_train.shape[0])):
    for word in data_train.loc[i,'intitle_words']:
        word_lib_bdctitle13.loc[word,data_train.loc[i,'category_new']]+=12
word_lib_bdctitle13_route = r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_df_bdctitle13.csv'
word_lib_bdctitle13.to_csv(word_lib_bdctitle13_route)
intitle_pathlist.append(word_lib_bdctitle13_route)
del word_lib_bdctitle13
word_lib_bdctitle14=word_lib_df.copy()
for i in tqdm(range(data_train.shape[0])):
    for word in data_train.loc[i,'intitle_words']:
        word_lib_bdctitle14.loc[word,data_train.loc[i,'category_new']]+=13
word_lib_bdctitle14_route = r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_df_bdctitle14.csv'
word_lib_bdctitle14.to_csv(word_lib_bdctitle14_route)
intitle_pathlist.append(word_lib_bdctitle14_route)
del word_lib_bdctitle14
word_lib_bdctitle15=word_lib_df.copy()
for i in tqdm(range(data_train.shape[0])):
    for word in data_train.loc[i,'intitle_words']:
        word_lib_bdctitle15.loc[word,data_train.loc[i,'category_new']]+=14
word_lib_bdctitle15_route = r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_df_bdctitle15.csv'
word_lib_bdctitle15.to_csv(word_lib_bdctitle15_route)
intitle_pathlist.append(word_lib_bdctitle15_route)
del word_lib_bdctitle15
word_lib_bdctitle16=word_lib_df.copy()
for i in tqdm(range(data_train.shape[0])):
    for word in data_train.loc[i,'intitle_words']:
        word_lib_bdctitle16.loc[word,data_train.loc[i,'category_new']]+=15
word_lib_bdctitle16_route = r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_df_bdctitle16.csv'
word_lib_bdctitle16.to_csv(word_lib_bdctitle16_route)
intitle_pathlist.append(word_lib_bdctitle16_route)
del word_lib_bdctitle16
word_lib_bdctitle17=word_lib_df.copy()
for i in tqdm(range(data_train.shape[0])):
    for word in data_train.loc[i,'intitle_words']:
        word_lib_bdctitle17.loc[word,data_train.loc[i,'category_new']]+=16
word_lib_bdctitle17_route = r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_df_bdctitle17.csv'
word_lib_bdctitle17.to_csv(word_lib_bdctitle17_route)
intitle_pathlist.append(word_lib_bdctitle17_route)
del word_lib_bdctitle17
word_lib_bdctitle18=word_lib_df.copy()
for i in tqdm(range(data_train.shape[0])):
    for word in data_train.loc[i,'intitle_words']:
        word_lib_bdctitle18.loc[word,data_train.loc[i,'category_new']]+=17
word_lib_bdctitle18_route = r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_df_bdctitle18.csv'
word_lib_bdctitle18.to_csv(word_lib_bdctitle18_route)
intitle_pathlist.append(word_lib_bdctitle18_route)
del word_lib_bdctitle18
word_lib_bdctitle19=word_lib_df.copy()
for i in tqdm(range(data_train.shape[0])):
    for word in data_train.loc[i,'intitle_words']:
        word_lib_bdctitle19.loc[word,data_train.loc[i,'category_new']]+=18
word_lib_bdctitle19_route = r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_df_bdctitle19.csv'
word_lib_bdctitle19.to_csv(word_lib_bdctitle19_route)
intitle_pathlist.append(word_lib_bdctitle19_route)
del word_lib_bdctitle19
word_lib_bdctitle20=word_lib_df.copy()
for i in tqdm(range(data_train.shape[0])):
    for word in data_train.loc[i,'intitle_words']:
        word_lib_bdctitle20.loc[word,data_train.loc[i,'category_new']]+=19
word_lib_bdctitle20_route = r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_df_bdctitle20.csv'
word_lib_bdctitle20.to_csv(word_lib_bdctitle20_route)
intitle_pathlist.append(word_lib_bdctitle20_route)
del word_lib_bdctitle20


for i in range(11,21):
    #tf_bdc
    word_lib_path=intitle_pathlist[i-2]
    train_feature_sparse_matrix, train_label_list = algo.tf_bdc(data_train,word_lib_path=word_lib_path)
    test_feature_sparse_matrix, test_label_list = algo.tf_bdc(data_test,word_lib_path=word_lib_path)
    predict_test_label_list = clf.SVM(train_feature_sparse_matrix, train_label_list, test_feature_sparse_matrix)
    Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
    tmp=pd.DataFrame(['tf_bdc_intitle',i,Macro_F1,Micro_F1]).T
    tmp.columns=results.columns
    results = results.append(tmp)
    word_lib_path=inabst_pathlist[i-2]
    train_feature_sparse_matrix, train_label_list = algo.tf_bdc(data_train,word_lib_path=word_lib_path)
    test_feature_sparse_matrix, test_label_list = algo.tf_bdc(data_test,word_lib_path=word_lib_path)
    predict_test_label_list = clf.SVM(train_feature_sparse_matrix, train_label_list, test_feature_sparse_matrix)
    Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
    tmp=pd.DataFrame(['tf_bdc_inabst',i,Macro_F1,Micro_F1]).T
    tmp.columns=results.columns
    results = results.append(tmp)
    #tf_dc
    word_lib_path=intitle_pathlist[i-2]
    train_feature_sparse_matrix, train_label_list = algo.tf_dc(data_train,word_lib_path=word_lib_path)
    test_feature_sparse_matrix, test_label_list = algo.tf_dc(data_test,word_lib_path=word_lib_path)
    predict_test_label_list = clf.SVM(train_feature_sparse_matrix, train_label_list, test_feature_sparse_matrix)
    Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
    tmp=pd.DataFrame(['tf_dc_intitle',i,Macro_F1,Micro_F1]).T
    tmp.columns=results.columns
    results = results.append(tmp)
    word_lib_path=inabst_pathlist[i-2]
    train_feature_sparse_matrix, train_label_list = algo.tf_dc(data_train,word_lib_path=word_lib_path)
    test_feature_sparse_matrix, test_label_list = algo.tf_dc(data_test,word_lib_path=word_lib_path)
    predict_test_label_list = clf.SVM(train_feature_sparse_matrix, train_label_list, test_feature_sparse_matrix)
    Macro_F1,Micro_F1=evaluate(test_label_list, predict_test_label_list)
    tmp=pd.DataFrame(['tf_dc_inabst',i,Macro_F1,Micro_F1]).T
    tmp.columns=results.columns
    results = results.append(tmp)




word_lib_df_abct3 = word_lib_df.copy()                

for i in tqdm(range(data_train.shape[0])):
    for word in data_train.loc[i,'inabst_words']:
        word_lib_df_abct3.loc[word,data_train.loc[i,'category_new']]+=2
word_lib_bdctitle_route = r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_df_abct3.csv'
word_lib_df_abct3.to_csv(word_lib_bdctitle_route)



#*****************************************************
#   计算论文报告相关指标

# 文档类别
data_train['category_new'].value_counts()
data_test['category_new'].value_counts()
# 文本标题平均字数，文章平均字数，分词后文章词语数目分布图
data['title_length'] = data['title'].apply(lambda f:len(f))
data['title_length'].max() #65
data['title_length'].min() #2
data['title_length'].mean() #25.5769

data['doc_length1'] = data['seg_list_rmstop'].apply(lambda f:len(f))
data['doc_length2'] = data['seg_list_rmstop_rmnum'].apply(lambda f:len(f[0]))
data['doc_length2'].max() #5111
data['doc_length2'].min() #2
data['doc_length2'].mean() #283.1688
doc_length = data['doc_length2'].tolist()

import matplotlib.pyplot as plt
plt.hist(doc_length1,bins=20,range=(0,1000))
plt.show()

import seaborn as sns

sns.set_style({'xtick.major.size':100.0})
sns.set()
f,ax = plt.subplots(figsize=(10,8))
plt.xticks(np.arange(0, 1500, 300))
ax = sns.countplot(data['doc_length2'])
ax.set_title("doc number distribution")
plt.xticks(np.linspace(0,1500,4))
plt.xlabel('doc length')
plt.xticks(np.arange(0, 1500, 300))
ax.set_xticks(np.arange(0,1500,300))
plt.axis([0, 1300, 0, 100])
ax.set_xticks([0,500,1000,1500])

plt.ylim(0,100)
ax=sns.distplot(data['doc_length2'], bins=100, kde=False)
ax.set_title("doc length distribution")
plt.xlabel('doc length')

ax = sns.kdeplot(data['doc_length2'])
ax.set_title("doc number distribution")

#tf-idf 与 加了标题权重后的关系
# =============================================================================
# 评估分类效果 及 错分情况
# =============================================================================
from sklearn import metrics
import numpy as np
import pandas as pd


#####把0-9映射回中文
#原始
test_label_list, predict_test_label_list = get_test_label_list(train_file_route, test_file_route, 'tf_idf_tftimes', 'SVM',title_times=1)
test_label_list =list(pd.Series( test_label_list).map({0:'保险', 1:'基金', 2:'外汇', 3:'期货', 4:'科技', 5:'股票',6:'银行债券',7:'黄金'}))
predict_test_label_list = list(pd.Series( predict_test_label_list).map({0:'保险', 1:'基金', 2:'外汇', 3:'期货', 4:'科技', 5:'股票',6:'银行债券',7:'黄金'}))
rs1=np.array(test_label_list)==np.array(predict_test_label_list)

test_label_list1, predict_test_label_list1 = get_test_label_list(train_file_route, test_file_route, 'tf_idf_tftimes', 'SVM',title_times=3)
test_label_list1 =list(pd.Series( test_label_list1).map({0:'保险', 1:'基金', 2:'外汇', 3:'期货', 4:'科技', 5:'股票',6:'银行债券',7:'黄金'}))
predict_test_label_list1 = list(pd.Series( predict_test_label_list1).map({0:'保险', 1:'基金', 2:'外汇', 3:'期货', 4:'科技', 5:'股票',6:'银行债券',7:'黄金'}))
rs2 = np.array(test_label_list1)==np.array(predict_test_label_list1)
rs3 = rs1==rs2

test_label_list2, predict_test_label_list2 = get_test_label_list(train_file_route, test_file_route, 'tf_idf_tftimes', 'SVM',abst_times=3,perc=0.1)
test_label_list2 =list(pd.Series( test_label_list2).map({0:'保险', 1:'基金', 2:'外汇', 3:'期货', 4:'科技', 5:'股票',6:'银行债券',7:'黄金'}))
predict_test_label_list2 = list(pd.Series( predict_test_label_list2).map({0:'保险', 1:'基金', 2:'外汇', 3:'期货', 4:'科技', 5:'股票',6:'银行债券',7:'黄金'}))
rs3 = np.array(test_label_list2)==np.array(predict_test_label_list2)


not_match = np.where(rs3==False)[0] #
wrong2right=[]
for i in not_match:
    if rs1[i]==False and rs2[i]==True:
        wrong2right.append(i)
title=data_test.loc[1637,'title']
cate = data_test.loc[1637,'category_new']
word_list=data_test.loc[1637,'seg_list_rmstop_rmnum']
print(word_list)
in_title_word=[]
for i in word_list:
    if i in title:
        in_title_word.append(i)
print(set(in_title_word))
doc_word_counter = Counter(word_list,sort=True)
sorted(doc_word_counter.items(),key=lambda pair:pair[1],reverse=True)
Counter.most_common(doc_word_counter,10)
count_dict = dict(Counter.most_common(doc_word_counter))
count_10_dict = dict(Counter.most_common(doc_word_counter,10)) 
for k,v in count_dict.items():
    if k in set(in_title_word):
        count_dict[k]=v*3
count_10_dict1=dict(Counter.most_common(count_dict,10))
# 还需要看构建的特征的值
inabst_words=word_list[:int(len(word_list)*0.1)]
count_dict_abst = dict(Counter.most_common(doc_word_counter))
for k,v in count_dict_abst.items():
    if k in set(inabst_words):
        count_dict_abst[k]=v*3
## 看看词频的分布
import data_util as du
word_lib_df = du.get_word_lib_df()  
word_freq_list= word_lib_df['total_freq'].tolist() 
ax = sns.countplot(word_lib_df['total_freq'])
ax.set_title("doc number distribution") 

#高频无意义的词
Counter.most_common(word_lib_df['total_freq'],20)
#低频词占比
ax=sns.distplot(word_lib_df['total_freq'], kde=False)
ax.set_title("word freq distribution")
plt.xlabel('word freq')
plt.xlim(0,100)
ax.set_xticks([0,100])
ax = sns.kdeplot(word_lib_df['total_freq'])
ax.set_title("word freq distribution")
plt.xlabel('word freq')
low_freq_w1 = word_lib_df[word_lib_df['total_freq']>4].index.tolist()
len(low_freq_w1)
low_freq_w_set = set(low_freq_w)


#####
# Do classification task, 
# then get the ground truth and the predict label named y_true and y_pred
classify_report = metrics.classification_report(test_label_list, predict_test_label_list)
confusion_matrix = metrics.confusion_matrix(test_label_list, predict_test_label_list)
overall_accuracy = metrics.accuracy_score(test_label_list, predict_test_label_list)
acc_for_each_class = metrics.precision_score(test_label_list, predict_test_label_list, average=None)
average_accuracy = np.mean(acc_for_each_class)
score = metrics.accuracy_score(test_label_list, predict_test_label_list)
print('classify_report : \n', classify_report)
print('confusion_matrix : \n', confusion_matrix)
print('acc_for_each_class : \n', acc_for_each_class)
print('average_accuracy: {0:f}'.format(average_accuracy))
print('overall_accuracy: {0:f}'.format(overall_accuracy))
print('score: {0:f}'.format(score))

# confusion matrix
import matplotlib
from sklearn.metrics import confusion_matrix
import itertools
zhfont1 = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\msyh.ttc')
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



#*****************************************************














   
# confusion matrix
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np
zhfont1 = matplotlib.font_manager.FontProperties(fname=r'C:\Users\DELL\Desktop\SIMYOU.TTF')
classes_txt     = ['保险', '债券', '基金', '外汇', '期货', '科技', '股票','银行','黄金']
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
plot_confusion_matrix(confusion_matrix(test_label_list, predict_test_label_list), classes_txt)
    