"""Converts LLM response into input for A2C model"""

import json 
import torch 
from Graph.data.community import CommunityFeatures 
from typing import Any , Dict , List , Tuple , Optional
from Graph.data.entity import Entity
from Graph.data.relationship import Relationship
from dataclasses import dataclass
from RL.environment import State , Action , Environment , Reward 
from model.llm import PaperRetrievalSystem 
import tiktoken
from transformers import AutoTokenizer , AutoModel
from Graph.data.community import CommunityScore
import re
import numpy as np 

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="original")


@dataclass
class TokenizeOutput: 
    text: str
    tokens: torch.Tensor
    attention_mask: torch.Tensor
    embeddings: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = None


@dataclass
class TensorizedFeatures:
    """Container for tensorized features extracted from text"""
    text_embeddings: torch.Tensor
    feature_vector: torch.Tensor
    categorical_features: Dict[str, torch.Tensor]
    metadata: Dict[str, Any]


class TextTokenizeOutput:

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()


        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def tokenize_text(self, text: str, max_length: int = 512) -> TokenizeOutput:
        """
        Tokenize text and create tensor representations
        
        Args:
            text: Input text to tokenize
            max_length: Maximum sequence length
            
        Returns:
            TokenizedOutput containing tokens, attention mask, and metadata
        """
        # Clean and preprocess text
        cleaned_text = self._preprocess_text(text)
        
        # Tokenize
        encoded = self.tokenizer(
            cleaned_text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            outputs = self.model(**encoded)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return TokenizeOutput(
            text=cleaned_text,
            tokens=encoded['input_ids'],
            attention_mask=encoded['attention_mask'],
            embeddings=embeddings,
            metadata={
                'original_length': len(text),
                'tokenized_length': encoded['input_ids'].shape[1],
                'truncated': len(text.split()) > max_length
            }
        )
    
    def extract_features_from_text(self, text: str) -> TensorizedFeatures:
        """
        Extract structured features from text and convert to tensors
        
        Args:
            text: Input text to process
            
        Returns:
            TensorizedFeatures containing various tensor representations
        """
        tokenized = self.tokenize_text(text)
        
        numerical_features = self._extract_numerical_features(text)
        
        categorical_features = self._extract_categorical_features(text)
        
        feature_vector = torch.tensor(numerical_features, dtype=torch.float32)
        
        categorical_tensors = {}
        for key, values in categorical_features.items():
            categorical_tensors[key] = self._categoricals_to_tensor(values)
        
        return TensorizedFeatures(
            text_embeddings=tokenized.embeddings,
            feature_vector=feature_vector,
            categorical_features=categorical_tensors,
            metadata={
                'text_length': len(text),
                'num_features': len(numerical_features),
                'categorical_counts': {k: len(v) for k, v in categorical_features.items()}
            }
        )
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for tokenization"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', ' ', text)
        return text.strip()
    
    def _extract_numerical_features(self, text: str) -> List[float]:
        """Extract numerical features from text"""
        features = []
        
        features.append(len(text))  
        features.append(len(text.split())) 
        features.append(len(text.split('.')))  
        
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if numbers:
            numbers = [float(n) for n in numbers if n.replace('.', '').replace('-', '').isdigit()]
            features.extend([
                len(numbers), 
                np.mean(numbers) if numbers else 0,  
                np.std(numbers) if len(numbers) > 1 else 0, 
                max(numbers) if numbers else 0,  
                min(numbers) if numbers else 0   
            ])
        else:
            features.extend([0, 0, 0, 0, 0])
        
        features.append(text.lower().count('community'))  
        features.append(text.lower().count('author'))     
        features.append(text.lower().count('paper'))      
        features.append(text.lower().count('citation'))  
        features.append(text.lower().count('relevant'))  
        
        return features
    
    def _extract_categorical_features(self, text: str) -> Dict[str, List[str]]:
        """Extract categorical features from text"""
        categorical = {
            'research_areas': [],
            'methods': [],
            'entities': [],
            'relationships': []
        }
        
        research_keywords = ['machine learning', 'deep learning', 'nlp', 'computer vision', 
                           'data mining', 'artificial intelligence', 'statistics', 'optimization']
        for keyword in research_keywords:
            if keyword.lower() in text.lower():
                categorical['research_areas'].append(keyword)
        
        method_keywords = ['classification', 'regression', 'clustering', 'neural network',
                         'svm', 'decision tree', 'random forest', 'gradient boosting']
        for method in method_keywords:
            if method.lower() in text.lower():
                categorical['methods'].append(method)
        
        entity_keywords = ['author', 'paper', 'institution', 'venue', 'concept']
        for entity in entity_keywords:
            if entity.lower() in text.lower():
                categorical['entities'].append(entity)
        
        relationship_keywords = ['collaboration', 'citation', 'co-authorship', 'affiliation']
        for rel in relationship_keywords:
            if rel.lower() in text.lower():
                categorical['relationships'].append(rel)
        
        return categorical
    
    def _categoricals_to_tensor(self, categories: List[str]) -> torch.Tensor:
        """Convert categorical features to tensor representation"""
        if not categories:
            return torch.zeros(1, dtype=torch.float32)
        
        unique_categories = list(set(categories))
        counts = [categories.count(cat) for cat in unique_categories]
        return torch.tensor(counts, dtype=torch.float32)
    
    def extract_json_features(self , text:str) -> List[CommunityScore]: 
        """For reward function: to take json values and then give into for trainin"""

        data = json.loads(text)
        out = []
        for row in data:
            out.append(
                CommunityScore(
                    community_id=row["community_id"],
                    relevance=float(row["relevance_score"]),
                    quality=float(row["quality_score"]),
                    diversity=float(row["diversity_score"]),
                    meta={k: v for k, v in row.items() if k not in {
                        "community_id","relevance_score","quality_score","diversity_score"
                    }},
                )
            )
        return out
    
    



    



     