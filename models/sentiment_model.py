import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig
import logging

logger = logging.getLogger(__name__)

class CustomerFeedbackClassifier(nn.Module):
   
    
    def __init__(self, model_name="distilbert-base-uncased", num_labels=4, dropout=0.1):
        super(CustomerFeedbackClassifier, self).__init__()
        
        self.config = DistilBertConfig.from_pretrained(model_name)
        self.distilbert = DistilBertModel.from_pretrained(model_name, config=self.config)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.dim, num_labels)
        self.num_labels = num_labels
        
        logger.info(f"Initialized model with {num_labels} output classes")
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        hidden_state = outputs[0]
        pooled_output = hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.hidden_states
        }
    
    def save_model(self, path):
        
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'num_labels': self.num_labels
        }, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load_model(cls, path, model_name="distilbert-base-uncased"):
       
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        model = cls(model_name=model_name, num_labels=checkpoint['num_labels'])
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {path}")
        return model
