import torch.nn as nn
from transformers import DistilBertModel

class DISTILBERTUncased(nn.Module):

    def __init__(self):
        
        super(DISTILBERTUncased, self).__init__()

        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask):
       
        bert_out = self.bert(input_ids, attention_mask, return_dict=False)[0][:,0]
        output = self.out(self.bert_drop(bert_out))
        
        return output