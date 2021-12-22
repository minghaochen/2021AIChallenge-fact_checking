from simpletransformers.language_representation import RepresentationModel
import pandas as pd
import gensim
from gensim.models import Word2Vec
from gensim.test.utils import common_texts
import nltk
from nltk.corpus import stopwords
from  nltk.stem import PorterStemmer
import re
import numpy as np
from simpletransformers.config.model_args import ModelArgs
import torch
import torch.nn as nn

############### zhaohui #####################
import faiss
from faiss import normalize_L2

evidence = pd.read_csv('evidence.csv',sep='\t')
fact_checking_train = pd.read_csv('fact_checking_train.csv',sep='\t')
fact_checking_test = pd.read_csv('fact_checking_test.csv',sep='\t')

label_2_v = {'pants-fire':0,'false':1,'barely-true':2,'half-true':3,'mostly-true':4,'true':5}
fact_checking_train['label'] = fact_checking_train['label'].map(label_2_v)

from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer()
X = vec.fit_transform(fact_checking_train['claim'])
Y = vec.transform(evidence['evidence'])
Z = vec.transform(fact_checking_test['claim'])
X = X.toarray().astype('float32')
Y = Y.toarray().astype('float32')
Z = Z.toarray().astype('float32')

normalize_L2(X)
normalize_L2(Y)
normalize_L2(Z)


k = 1  # 定义召回向量个数
index=faiss.IndexFlatIP(16123) # 点乘，归一化的向量点乘即cosine相似度（越大越好）
index.add(Y) # 添加训练时的样本
D, I1 = index.search(X, k) # 寻找相似向量， I表示相似用户ID矩阵， D表示距离矩阵
D, I2 = index.search(Z, k) # 寻找相似向量， I表示相似用户ID矩阵， D表示距离矩阵
##########################################################

from sklearn.decomposition import TruncatedSVD
svdT = TruncatedSVD(n_components=768,random_state=2021)
svdT.fit(X)

X = svdT.transform(X)
np.save('train_vectors_3.npy', X)
Y = svdT.transform(Y)
np.save('evidence_vectors_3.npy', Y)



evidence = pd.read_csv('evidence.csv',sep='\t')
evidence['label']=0
fact_checking_train = pd.read_csv('fact_checking_train.csv',sep='\t')
fact_checking_test = pd.read_csv('fact_checking_test.csv',sep='\t')
fact_checking_test['label']=0
print(evidence.shape)
print(fact_checking_train.shape)
print(fact_checking_test.shape)

label_2_v = {'pants-fire':0,'false':1,'barely-true':2,'half-true':3,'mostly-true':4,'true':5}
weight = 4462*torch.tensor([1/2233,1/4462,1/2980,1/3171,1/2898,1/2256])
fact_checking_train['label'] = fact_checking_train['label'].map(label_2_v)
# train_y = fact_checking_train['label'].values
#
#
# train = fact_checking_train[['claim','label']].copy()
# train.columns = ["text", "labels"]
# test = fact_checking_test[['claim','label']].copy()
# test.columns = ["text", "labels"]
# train = pd.concat([train,test]).reset_index(drop=True)
# print(train.shape)



# model_args = ModelArgs(max_seq_length=256)
# model = RepresentationModel(
#         model_type="roberta",
#         model_name="/data1/zhengjl/Pycharm/fact_checking/models/roberta-base",
#         cuda_device=1,
#         args=model_args,
#     )

from tqdm import tqdm

############## 获得embedding
# evidence_vectors = []
# for idx, row in tqdm(evidence.iterrows()):
#     sentence_list = [row['evidence']]
#     vec = model.encode_sentences(sentence_list)
#     if vec.shape[1] < 256:
#         vec = np.concatenate([vec,np.zeros((1,256-vec.shape[1],768),dtype=np.float32)],axis=1)
#     evidence_vectors.append(
#         vec
#     )
# evidence_vectors = np.vstack(evidence_vectors)
# print(evidence_vectors.shape)
# np.save('evidence_vectors.npy', evidence_vectors)

# train_vectors = []
# for idx, row in tqdm(fact_checking_train.iterrows()):
#     sentence_list = [row['claim']]
#     vec = model.encode_sentences(sentence_list)
#     if vec.shape[1] < 256:
#         vec = np.concatenate([vec,np.zeros((1,256-vec.shape[1],768),dtype=np.float32)],axis=1)
#     train_vectors.append(
#         vec
#     )
# train_vectors = np.vstack(train_vectors)
# print(train_vectors.shape)
# np.save('train_vectors.npy', train_vectors)
# print(train_vectors.shape)

# test_vectors = []
# for idx, row in tqdm(fact_checking_test.iterrows()):
#     sentence_list = [row['claim']]
#     vec = model.encode_sentences(sentence_list)
#     if vec.shape[1] < 256:
#         vec = np.concatenate([vec,np.zeros((1,256-vec.shape[1],768),dtype=np.float32)],axis=1)
#     test_vectors.append(
#         vec
#     )
# test_vectors = np.vstack(test_vectors)
# print(test_vectors.shape)
# np.save('test_vectors.npy', test_vectors)
# print(test_vectors.shape)
############################ 获得句子embedding
# print('train')
# sentences = fact_checking_train['claim'].values.tolist()
# train_vectors = model.encode_sentences(sentences, combine_strategy="mean")
# np.save('train_vectors_2.npy', train_vectors)
# print('evidence')
# sentences = evidence['evidence'].values.tolist()
# evidence_vectors = model.encode_sentences(sentences, combine_strategy="mean")
# print('over')
# np.save('evidence_vectors_2.npy', evidence_vectors)

class AEModel(nn.Module):
    def __init__(self):
        super(AEModel, self).__init__()
        # self.multihead_attn = nn.MultiheadAttention(768, 8)
        # self.multihead_attn2 = nn.MultiheadAttention(768, 8)
        # self.encoder = nn.TransformerEncoderLayer(768,8)
        self.MLP1 = nn.Sequential(
            nn.Linear(in_features=768*2, out_features=768, bias=True),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=768, out_features=768, bias=True),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=768, out_features=768, bias=True)
        )

        self.MLP2 = nn.Sequential(
            nn.Linear(in_features=768*2, out_features=512, bias=True),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=512, out_features=128, bias=True),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=128, out_features=64, bias=True),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=64, out_features=6, bias=True)
        )
        # for p in self.MLP1.parameters():
        #     if p.dim() > 1:
        #         nn.init.xavier_uniform_(p)
        # for p in self.MLP2.parameters():
        #     if p.dim() > 1:
        #         nn.init.xavier_uniform_(p)

    def forward(self, x ,y):
        y = self.MLP1(y)
        x = torch.cat((x,y),1)
        x = self.MLP2(x)
        return x



# model = AEModel()
# q = torch.randn(256, 1024, 768)
# kv = torch.randn(256*5, 1024, 768)
#
# output = model(q,kv,kv)
# print(output.shape)

import sklearn
from sklearn.model_selection import KFold, StratifiedKFold
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1996)

evidence_vectors = np.load('evidence_vectors_2.npy')
train_vectors = np.load('train_vectors_2.npy')


# evidence_vectors = torch.from_numpy(evidence_vectors).float()
train_vectors = torch.from_numpy(train_vectors).float()

from sklearn.metrics import accuracy_score

for train_index, test_index in kf.split(fact_checking_train, fact_checking_train['label']):


    model = AEModel().to('cuda:1')
    criterion = nn.CrossEntropyLoss(weight=weight.to('cuda:1'))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    train_loader = torch.utils.data.DataLoader(
        train_index, batch_size=1024,
        shuffle=True, drop_last=False)
    valid_loader = torch.utils.data.DataLoader(
        test_index, batch_size=1024,
        shuffle=False, drop_last=False)


    for epoch in range(300):
        model.train()
        for batch in tqdm(train_loader):

            train_batch = train_vectors[batch].to('cuda:1')
            evidence_batch = evidence_vectors[I1[batch]]
            evidence_batch = torch.from_numpy(evidence_batch).float().squeeze(1).to('cuda:1')
            output = model(train_batch, torch.cat((train_batch,evidence_batch),1))

            target = torch.from_numpy(fact_checking_train['label'].iloc[batch].values).to('cuda:1', dtype=torch.int64)
            loss = criterion(output, target)
            optimizer.zero_grad()  # clear gradients for next train
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

        model.eval()
        y_pred = []
        with torch.no_grad():
            for batch in tqdm(train_loader):
                train_batch = train_vectors[batch].to('cuda:1')
                evidence_batch = evidence_vectors[I1[batch]]
                evidence_batch = torch.from_numpy(evidence_batch).float().squeeze(1).to('cuda:1')
                output = model(train_batch, torch.cat((train_batch, evidence_batch), 1))
                y_pred.append(output.cpu().numpy())

        y_pred = np.vstack(y_pred)
        # print(y_pred)
        y_pred = np.argmax(y_pred, axis=1)

        acc = accuracy_score(fact_checking_train['label'].iloc[train_index], y_pred)
        print('train',acc)


        model.eval()
        y_pred = []
        with torch.no_grad():
            for batch in tqdm(valid_loader):
                train_batch = train_vectors[batch].to('cuda:1')
                evidence_batch = evidence_vectors[I1[batch]]
                evidence_batch = torch.from_numpy(evidence_batch).float().squeeze(1).to('cuda:1')
                output = model(train_batch, torch.cat((train_batch, evidence_batch), 1))
                y_pred.append(output.cpu().numpy())


        y_pred = np.vstack(y_pred)
        # print(y_pred)
        y_pred = np.argmax(y_pred, axis=1)

        acc = accuracy_score(fact_checking_train['label'].iloc[test_index], y_pred)
        print('valid',acc)

    break



