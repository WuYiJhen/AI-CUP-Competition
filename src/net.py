import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, BertConfig
PRETRAINED_MODEL_NAME = 'allenai/scibert_scivocab_uncased'


class swish(nn.Module):
    def __init__(self):
        super(swish, self).__init__()

    def forward(self, x):
        x = x * nn.Sigmoid()(x)
        return x


class BertforMultilabel(BertPreTrainedModel):
    def __init__(self, conf, num_class, dropout=0.2):
        super(BertforMultilabel, self).__init__(conf)
        self.Y = num_class
        self.dropout = dropout
        self.embedding_dim = conf.hidden_size
        self.bert = BertModel.from_pretrained(PRETRAINED_MODEL_NAME)
        self.dropout = nn.Dropout(p=dropout)
        self.hidden1 = nn.Linear(self.embedding_dim, self.embedding_dim//2)
        self.final = nn.Linear(self.embedding_dim//2, self.Y)

    def key_padding_mask_bert(self, x, pad_num=0):
        atten_mask = x.masked_fill(x != pad_num, 1)
        return atten_mask

    def forward(self, batch):
        _, x = self.bert(input_ids=batch.content,
                         attention_mask=self.key_padding_mask_bert(batch.content))
        x = self.dropout(x)
        x = swish()(self.hidden1(x))
        x = self.dropout(x)
        x = self.final(x)
        return x
