import re
import string
from transformers import DistilBertTokenizer
import logging

logger = logging.getLogger(__name__)

class TextPreprocessor:
    """Preprocess text data for model input"""
    
    def __init__(self, model_name="distilbert-base-uncased"):
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        logger.info(f"Initialized tokenizer for {model_name}")
    
    def clean_text(self, text):
        """Basic text cleaning"""
        if not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove punctuation (optional, BERT handles punctuation well)
        # text = text.translate(str.maketrans('', '', string.punctuation))
        
        return text
    
    def preprocess_batch(self, texts):
        """Preprocess a batch of texts"""
        cleaned_texts = [self.clean_text(text) for text in texts]
        return cleaned_texts
    
    def tokenize_text(self, text, max_length=256):
        """Tokenize single text for model input"""
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        return encoding
    
    def tokenize_batch(self, texts, max_length=256):
        """Tokenize batch of texts"""
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        )
        return encodings