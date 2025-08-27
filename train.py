import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import yaml
import logging
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger(__name__)

class FeedbackDataset(Dataset):
    """PyTorch Dataset for customer feedback data"""
    
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class DataProcessor:
    """Process and load customer feedback data"""
    
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.categories = self.config['data']['categories']
        self.category_to_id = {cat: idx for idx, cat in enumerate(self.categories)}
        self.id_to_category = {idx: cat for idx, cat in enumerate(self.categories)}
    
    def load_sample_data(self):
        """Load sample data for demonstration"""
        # In a real scenario, this would load from database/API
        sample_data = [
            {"text": "The app is amazing! Very user-friendly.", "category": "positive"},
            {"text": "I can't login to my account, need urgent help!", "category": "urgent"},
            {"text": "Please add dark mode feature", "category": "feature_request"},
            {"text": "The service is terrible and slow", "category": "negative"},
            {"text": "Billing issue: charged twice", "category": "negative"},
            {"text": "Love the new update! Great work team!", "category": "positive"},
            {"text": "System crashed and lost my data", "category": "urgent"},
            {"text": "Would be great to have export functionality", "category": "feature_request"}
        ]
        
        texts = [item['text'] for item in sample_data]
        labels = [self.category_to_id[item['category']] for item in sample_data]
        
        return texts, labels
    
    def create_data_loaders(self, tokenizer, batch_size=16):
        """Create train, validation, and test data loaders"""
        texts, labels = self.load_sample_data()
        
        # Split data
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            texts, labels, test_size=0.3, random_state=42, stratify=labels
        )
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
        )
        
        # Create datasets
        train_dataset = FeedbackDataset(train_texts, train_labels, tokenizer)
        val_dataset = FeedbackDataset(val_texts, val_labels, tokenizer)
        test_dataset = FeedbackDataset(test_texts, test_labels, tokenizer)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        logger.info(f"Created data loaders: Train={len(train_loader)}, Val={len(val_loader)}, Test={len(test_loader)}")
        
        return train_loader, val_loader, test_loader