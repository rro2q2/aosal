# Author: Roland Oruche
# Affiliation: University of Missouri-Columbia
# Year: 2024

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel, PretrainedConfig, BertConfig, BertForSequenceClassification

class BertModel():
    """ BERT module.
    Functions:
        forward(): Performs forward pass using `BertForSequenceClassification` class. 
    """
    def __init__(
        self,
        num_labels: int 
    ):
        super().__init__()
        self.num_labels = num_labels

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=self.num_labels)
        self.bert.config.output_hidden_states = True
        self.bert.to(device)

    def forward(
        self, 
        input_ids=None,
        attention_mask=None,
        labels=None
    ) -> Tuple[torch.FloatTensor]:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return outputs
    
if __name__ == '__main__':
    _ = BertModel()
