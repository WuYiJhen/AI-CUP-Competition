import pandas as pd
import numpy as np
import re
import string
import pickle
from sklearn.model_selection import train_test_split


def strQ2B(s):
    n = []
    for char in str(s):
        num = ord(char)
        if num == 0x3000:
            num = 32
        elif 0xFF01 <= num <= 0xFF5E:
            num -= 0xfee0
        num = chr(num)
        n.append(num)
    return ''.join(n)


remove_sss = re.compile(r'(\$\$\$)')


def preprocess(x):
    x = remove_sss.sub(' ', x)
    x = strQ2B(x)
    return x.lower()


trainset = pd.read_csv('./data/trainset.csv')
testset = pd.read_csv('./data/testset.csv')

trainset.rename(columns={'Abstract': 'content', 'Title': 'headline',
                         'Classifications': 'label'}, inplace=True)
testset.rename(columns={'Abstract': 'content', 'Title': 'headline',
                        'Classifications': 'label'}, inplace=True)

trainset['headline'] = trainset['headline'].apply(preprocess)
trainset['content'] = trainset['content'].apply(preprocess)

testset['headline'] = testset['headline'].apply(preprocess)
testset['content'] = testset['content'].apply(preprocess)

trainset['content'] = trainset['headline'] + '. ' + trainset['content']
testset['content'] = testset['headline'] + '. ' + testset['content']

train_data, valid_data = train_test_split(
    trainset, test_size=0.1, random_state=42)
train_data = train_data.reset_index(drop=True)
valid_data = valid_data.reset_index(drop=True)
train_data.to_csv('./data/train_data.csv', index=False)
valid_data.to_csv('./data/valid_data.csv', index=False)
testset.to_csv('./data/test_data.csv', index=False)

label2int = {'ENGINEERING': 0, 'EMPIRICAL': 1, 'THEORETICAL': 2, 'OTHERS': 3}
with open('./token_info/label2int.pickle', 'wb') as output:
    pickle.dump(label2int, output)
output.close()
