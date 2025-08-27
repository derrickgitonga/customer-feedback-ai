import torch
import yaml
import pandas as pd
from typing import List, Dict, Any
import json

from models.sentiment_model import CustomerFeedbackClassifier
from models.theme_extractor import ThemeExtractor
from utils.preprocessor import TextPreprocessor
from utils.logger import setup_logging

logger = setup_logging()

class FeedbackAnalyzer:
    """Main inference class for customer feedback analysis"""
    
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        model_path = f"{self.config['training']['save_dir']}/best_model.pth"
        self.model = CustomerFeedbackClassifier.load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load tokenizer and theme extractor
        self.tokenizer = TextPreprocessor().tokenizer
        self.theme_extractor = ThemeExtractor(config_path)
        self.categories = self.config['data']['categories']
        
        logger.info("Feedback analyzer initialized successfully")
    
    def analyze_single(self, text: str) -> Dict[str, Any]:
        """Analyze single piece of feedback"""
        # Preprocess and tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.config['model']['max_length'],
            return_tensors='pt'
        )
        
        # Predict
        with torch.no_grad():
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            outputs = self.model(input_ids, attention_mask)
            probabilities = torch.softmax(outputs['logits'], dim=1)
            confidence, prediction = torch.max(probabilities, 1)
        
        # Extract themes
        themes = self.theme_extractor.extract_themes(text)
        
        return {
            'text': text,
            'category': self.categories[prediction.item()],
            'confidence': confidence.item(),
            'themes': themes,
            'probabilities': {cat: prob.item() for cat, prob in zip(self.categories, probabilities[0])}
        }
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Analyze multiple pieces of feedback"""
        results = []
        for text in texts:
            results.append(self.analyze_single(text))
        return results
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """Analyze feedback from pandas DataFrame"""
        results = self.analyze_batch(df[text_column].tolist())
        
        # Convert to DataFrame
        result_df = pd.DataFrame(results)
        
        # Add original data
        for col in df.columns:
            if col != text_column:
                result_df[col] = df[col].values
        
        return result_df
    
    def export_results(self, results: List[Dict[str, Any]], format: str = 'json'):
        """Export analysis results in specified format"""
        if format == 'json':
            return json.dumps(results, indent=2)
        elif format == 'csv':
            df = pd.DataFrame(results)
            return df.to_csv(index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

def main():
    """Example usage"""
    analyzer = FeedbackAnalyzer()
    
    # Example feedback
    sample_feedback = [
        "The app is amazing! Very user-friendly interface.",
        "I can't login to my account, need urgent help!",
        "Please add dark mode feature, it would be great for night usage.",
        "The service is terrible and slow, very disappointed."
    ]
    
    # Analyze batch
    results = analyzer.analyze_batch(sample_feedback)
    
    print("Feedback Analysis Results:")
    for result in results:
        print(f"\nText: {result['text']}")
        print(f"Category: {result['category']} (Confidence: {result['confidence']:.3f})")
        print(f"Themes: {result['themes']}")

if __name__ == "__main__":
    main()