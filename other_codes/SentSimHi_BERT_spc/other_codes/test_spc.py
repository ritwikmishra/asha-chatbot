
'''
testing our SPC on dpil dataset
'''

from transformers import  AutoTokenizer, AutoModel
from tqdm import tqdm
import pandas as pd, numpy as np
from utils import myBert
import torch

test_file_name = 'dpil_test_hi.csv'

hyper_params = {
    "run_ID": 126,
    "message": "124 with different seed",
    "createData": True,
    "bs": 32,
    "BERT_FROZEN": False,
    "bert_model_name": "bert-base-multilingual-cased",
    "model_options": "ai4bharat/indic-bert, bert-base-multilingual-cased, xlm-roberta-base, bert-base-multilingual-uncased, xlm-mlm-100-1280, xlm-mlm-tlm-xnli15-1024, xlm-mlm-xnli15-1024",
    "alpha": 1e-05,
    "epochs": 5,
    "rseed": 121,
    "nw": 8,
    "cut_off": 0.51,
    "label_smoothing": False,
    "random_negative_sent": True,
    "train_ratio": 0.5,
    "val_ratio": 0.2,
    "max_len": 300,
    "data_source": "inshorts_full",
    "data_options": "asha_sentsim, inshorts_full, inshorts_400pairs",
    "architecture": "fc3",
    "arch_options": "SequenceClassification, fc1, fc2, fc3",
    "which_bert_embedding": "pooler_output",
    "bert_embedding_options": "cls, pooler_output",
    "my_datapath": "data/",
    "criterion": "BCE",
    "query_expansion": False,
    "embedding_size": 768,
	'inflate_negative_n_times': 0,
    'include_curated_negatives': False
}

which_criterion = {
    'MSE': torch.nn.MSELoss(),
    'NLL': torch.nn.NLLLoss(),
    'CrossEntropy': torch.nn.CrossEntropyLoss(),
    'HingeEmbeddingLoss': torch.nn.HingeEmbeddingLoss(),
    'BCE': torch.nn.BCELoss(),
    'BCEwithLogits': torch.nn.BCEWithLogitsLoss()
}

mn = 'BERT_spc'
suff = ''
checkpointname = '695_epoch_3'
print(checkpointname)
import torch
is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda:0")
    t = torch.cuda.get_device_properties(device).total_memory
    c = torch.cuda.memory_reserved(device)
    a = torch.cuda.memory_allocated(device)
    f = t -(c-a)  # free inside cache
    print("GPU is available", torch.cuda.get_device_name(), round(t/(1024*1024*1024)), "GB")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

mymodelST = myBert(device,hyper_params,which_criterion).to(device)
print("\n\t\t==== Model created  =====")
checkpoint = torch.load(checkpointname+'.pth.tar',map_location=torch.device(device)) if mn in ['BERT_spc','E'] else ''
print("\n\t\t==== Checkpoint loaded  =====")
print(mymodelST.load_state_dict(checkpoint['state_dict'])) if mn in ['BERT_spc','E'] else ''
del checkpoint

tokenizer = AutoTokenizer.from_pretrained(hyper_params['bert_model_name'])
test_data = pd.read_csv(test_file_name)
test_data['y'] = test_data['y'].apply(lambda d: 0.0 if d=='NP' else 1.0)
y_pred, y_true = [], []

for i, row in tqdm(test_data.iterrows(),ncols=100,total=len(test_data)):
    y_pred.append(0.0 if mymodelST.predict(row['Sent1'],row['Sent2'],tokenizer) < 0.5 else 1.0)
    y_true.append(row['y'])

y_pred = np.array(y_pred)
y_true = np.array(y_true)

assert len(y_pred) == len(y_true)

print('Accuracy', round(100*sum(y_pred == y_true)/len(y_pred), 3))


