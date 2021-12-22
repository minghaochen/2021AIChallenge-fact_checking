import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
device = torch.device("cpu")
import warnings
warnings.simplefilter('ignore')
from sklearn.model_selection import KFold, StratifiedKFold
from scipy.special import softmax
import sklearn
from sklearn.metrics import log_loss, f1_score

def seed_all(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

seed_all(79)

evidence = pd.read_csv('evidence.csv',sep='\t')
fact_checking_train = pd.read_csv('fact_checking_train.csv',sep='\t')
fact_checking_test = pd.read_csv('fact_checking_test.csv',sep='\t')
fact_checking_test['label']=0

# label_2_v = {'false':0,'half-true':1,'barely-true':2,'mostly-true':3,'true':4,'pants-fire':5}
label_2_v = {'pants-fire':0,'false':1,'barely-true':2,'half-true':3,'mostly-true':4,'true':5}
fact_checking_train['label'] = fact_checking_train['label'].map(label_2_v)

from simpletransformers.classification import ClassificationModel

# model configuration
model_args = {
    "save_eval_checkpoints": False,
    "save_model_every_epoch": False,
    'reprocess_input_data': True,
    'overwrite_output_dir': True,
    'manual_seed': 79,
    "silent": True,
    'num_train_epochs': 3,
    'learning_rate': 2e-5,
    'fp16': False,
    'max_seq_length': 256,
    'train_batch_size': 8,
    'eval_batch_size': 8,
    # 'cache_dir': 'models',
    # 'n_gpu' : 4,
#     'train_batch_size': 1,
}

# Preparing train data
train = fact_checking_train[['claim','label']].copy()
train.columns = ["text", "labels"]

test = fact_checking_test[['claim','label']].copy()
test.columns = ["text", "labels"]


torch.cuda.empty_cache()
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=79)
err=[]
y_pred_tot=[]
for train_index, test_index in kf.split(train, train['labels']):
    train1_trn, train1_val = train.iloc[train_index], train.iloc[test_index]
    model_rb = ClassificationModel('roberta', '/data1/zhengjl/Pycharm/fact_checking/models/roberta-base', num_labels=6,  args=model_args)
    model_rb.train_model(train1_trn, eval_df=train1_val)
    result, model_outputs, _ = model_rb.eval_model(train1_val, acc=sklearn.metrics.accuracy_score)
    print(f"Accuracy:{result['acc']}")
    err.append((result['acc']))
    predictions, raw_outputs  = model_rb.predict(test['text'].values.tolist())
    y_pred_tot.append(raw_outputs)
    # break
print("Mean f1 score: ",np.mean(err))
# np.save('y_pred_tot',y_pred_tot)