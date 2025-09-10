import re
from collections import Counter
import yaml
import logging

logger = logging.getLogger(__name__)

class ThemeExtractor:
    
    
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.theme_keywords = config['data']['theme_keywords']
        logger.info("Theme extractor initialized with %d theme categories", len(self.theme_keywords))
    
    def extract_themes(self, text):
       
        text = text.lower()
        themes = []
        
        for theme, keywords in self.theme_keywords.items():
            for keyword in keywords:
                if re.search(r'\b' + re.escape(keyword) + r'\b', text):
                    themes.append(theme)
                    break
        
        return list(set(themes))
    
    def batch_extract_themes(self, texts):
        
        results = []
        for text in texts:
            results.append(self.extract_themes(text))
        return results
    
    def get_theme_stats(self, texts):
        """Get statistics about theme frequency"""
        all_themes = []
        for text in texts:
            all_themes.extend(self.extract_themes(text))
        
        theme_counts = Counter(all_themes)
        return dict(theme_counts)
