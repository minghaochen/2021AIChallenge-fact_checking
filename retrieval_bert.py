import warnings
warnings.filterwarnings("ignore")
import faiss
from faiss import normalize_L2
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


# model = RepresentationModel(
#         model_type="roberta",
#         model_name="/data1/zhengjl/Pycharm/fact_checking/models/roberta-base",
#     )


# evidence = pd.read_csv('evidence.csv',sep='\t')
# sentence_list  = evidence['evidence'].values.tolist()
# evidence_vectors = model.encode_sentences(sentence_list)
# print(evidence_vectors.shape)

# fact_checking_train = pd.read_csv('fact_checking_train.csv',sep='\t')
# sentence_list  = fact_checking_train['claim'].values.tolist()
# train_vectors = model.encode_sentences(sentence_list)
# print(train_vectors.shape)


# evidence_vectors = evidence_vectors.astype('float32')
# train_vectors = train_vectors.astype('float32')
#
# normalize_L2(evidence_vectors)
# normalize_L2(train_vectors)
#
# k = 5  # 定义召回向量个数
# index=faiss.IndexFlatIP(768) # 点乘，归一化的向量点乘即cosine相似度（越大越好）
# index.add(evidence_vectors) # 添加训练时的样本
# D, I = index.search(train_vectors, k) # 寻找相似向量， I表示相似用户ID矩阵， D表示距离矩阵
# print(I)
# np.save('retrieval_train.npy', I)

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

k = 5  # 定义召回向量个数
index=faiss.IndexFlatIP(16123) # 点乘，归一化的向量点乘即cosine相似度（越大越好）
index.add(Y) # 添加训练时的样本
D, I1 = index.search(X, k) # 寻找相似向量， I表示相似用户ID矩阵， D表示距离矩阵
# print(I1)

D, I2 = index.search(Z, k) # 寻找相似向量， I表示相似用户ID矩阵， D表示距离矩阵
# print(I2)


from simpletransformers.classification import ClassificationModel

# model configuration
model_args = {
    "save_eval_checkpoints": False,
    "save_model_every_epoch": False,
    'reprocess_input_data': True,
    'overwrite_output_dir': True,
    'manual_seed': 1996,
    "silent": True,
    'num_train_epochs': 3,
    'learning_rate': 2e-5,
    'fp16': False,
    'max_seq_length': 256,
#     'train_batch_size': 1,
}

import sklearn
from sklearn.model_selection import KFold, StratifiedKFold
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1996)
y_pred_tot = []
err=[]
for topk in range(5):
    fold = 0
    for train_index, test_index in kf.split(fact_checking_train, fact_checking_train['label']):
        train1_trn, train1_val = fact_checking_train.iloc[train_index], fact_checking_train.iloc[test_index]

        train_data = []
        for idx,row in train1_trn.iterrows():
            train_data.append(
                [
                    row['claim'],
                    evidence.iloc[I1[idx][topk]]['evidence'],
                    row['label']
                 ]
            )
        train_df = pd.DataFrame(train_data)
        train_df.columns = ["text_a", "text_b", "labels"]


        eval_data  = []
        for idx, row in train1_val.iterrows():
            eval_data.append(
                [
                    row['claim'],
                    evidence.iloc[I1[idx][topk]]['evidence'],
                    row['label']
                ]
            )
        eval_df  = pd.DataFrame(eval_data)
        eval_df.columns = ["text_a", "text_b", "labels"]
        # print(eval_df.shape)

        model_rb = ClassificationModel('roberta', '/data1/zhengjl/Pycharm/fact_checking/models/roberta-base',
                                       num_labels=6, args=model_args, cuda_device=2)

        model_rb.train_model(train_df)
        result, model_outputs, _ = model_rb.eval_model(eval_df, acc=sklearn.metrics.accuracy_score)
        print(f"Accuracy:{result['acc']}")
        err.append((result['acc']))


        test_data = []
        for idx, row in fact_checking_test.iterrows():
            test_data.append(
                [
                    row['claim'],
                    evidence.iloc[I2[idx][topk]]['evidence'],
                ]
            )


        predictions, raw_outputs = model_rb.predict(test_data)
        # print(fact_checking_test.shape)
        # print(raw_outputs.shape)
        acc = result['acc']
        np.save(f'raw_outputs_roberta_1996_{topk}_{fold}_{acc}', raw_outputs)
        fold += 1
        # y_pred_tot.append(raw_outputs)

print("Mean f1 score: ",np.mean(err))
# np.save('y_pred_tot',y_pred_tot)


