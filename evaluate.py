import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from models.sentiment_model import CustomerFeedbackClassifier
from utils.data_loader import DataProcessor
from utils.preprocessor import TextPreprocessor
from utils.logger import setup_logging

logger = setup_logging()

class Evaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_processor = DataProcessor(config_path)
        
        # Load model
        model_path = os.path.join(self.config['training']['save_dir'], "best_model.pth")
        self.model = CustomerFeedbackClassifier.load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load tokenizer
        tokenizer_path = os.path.join(self.config['training']['save_dir'], "tokenizer")
        self.tokenizer = TextPreprocessor().tokenizer
    
    def comprehensive_evaluation(self):
        """Run comprehensive evaluation"""
        logger.info("Starting comprehensive evaluation...")
        
        # Get test data
        _, _, test_loader = self.data_processor.create_data_loaders(
            self.tokenizer,
            batch_size=self.config['model']['batch_size']
        )
        
        # Get predictions
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                probabilities = torch.softmax(outputs['logits'], dim=1)
                _, predictions = torch.max(outputs['logits'], 1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        self._generate_classification_report(all_labels, all_predictions)
        self._generate_confusion_matrix(all_labels, all_predictions)
        self._calculate_business_metrics(all_labels, all_predictions)
        
        logger.info("Evaluation completed!")
    
    def _generate_classification_report(self, true_labels, predictions):
        """Generate detailed classification report"""
        target_names = self.config['data']['categories']
        
        report = classification_report(
            true_labels, predictions, 
            target_names=target_names, 
            output_dict=True
        )
        
        logger.info("\nClassification Report:")
        logger.info(f"Overall Accuracy: {report['accuracy']:.4f}")
        
        for category in target_names:
            logger.info(f"{category}: Precision={report[category]['precision']:.4f}, "
                       f"Recall={report[category]['recall']:.4f}, "
                       f"F1={report[category]['f1-score']:.4f}")
    
    def _generate_confusion_matrix(self, true_labels, predictions):
        """Generate and save confusion matrix"""
        cm = confusion_matrix(true_labels, predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.config['data']['categories'],
                   yticklabels=self.config['data']['categories'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save plot
        plots_dir = "./evaluation_plots"
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        
        plt.savefig(os.path.join(plots_dir, "confusion_matrix.png"))
        plt.close()
        
        logger.info("Confusion matrix saved to evaluation_plots/confusion_matrix.png")
    
    def _calculate_business_metrics(self, true_labels, predictions):
        """Calculate business-relevant metrics"""
        # Urgent detection rate (if urgent is class 2)
        urgent_idx = self.config['data']['categories'].index('urgent')
        urgent_mask = np.array(true_labels) == urgent_idx
        urgent_detected = np.array(predictions)[urgent_mask] == urgent_idx
        
        urgent_detection_rate = np.mean(urgent_detected) if len(urgent_detected) > 0 else 0
        
        logger.info(f"Urgent Issue Detection Rate: {urgent_detection_rate:.4f}")
        logger.info(f"Missed Urgent Issues: {np.sum(~urgent_detected)}")

def main():
    """Main evaluation function"""
    try:
        evaluator = Evaluator()
        evaluator.comprehensive_evaluation()
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()