import pandas as pd


def get_data(file_route):
    with open(file_route,encoding='utf-8') as f:
        data = f.read()
    return data


def get_word_lib_df(word_lib_path=r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib_rmlowfreq_1221.csv'):
    word_lib_df = pd.read_csv(word_lib_path,encoding='utf-8',sep=',',engine='python')
    word_lib_df.set_index('word', inplace=True)
    return word_lib_df

#word_lib_df = pd.read_csv(r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib.csv',encoding='utf-8',sep=',',engine='python')
#word_lib_df['word'] = word_lib_df.index
#word_lib_df['len'] = word_lib_df['word'].apply(lambda x:len(str(x)))
#word_lib_df1 = word_lib_df.loc[word_lib_df['len']>=2]
#word_lib_df1.to_csv(r'C:\Users\DELL\Desktop\Courses\研究生\文本挖掘-王刚\实验报告\word_lib1.csv')
