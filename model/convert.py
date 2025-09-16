"""Converts LLM JSON response into input for A2C model"""

import json 
import torch 
from Graph.data.community import CommunityFeatures 
from typing import Any, Dict, List, Tuple, Optional
from Graph.data.entity import Entity
from Graph.data.relationship import Relationship
from dataclasses import dataclass
from RL.environment import State, Action, Environment, Reward 
from model.llm import PaperRetrievalSystem 
import tiktoken
from transformers import AutoTokenizer, AutoModel
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
    """Container for tensorized features extracted from JSON data"""
    text_embeddings: torch.Tensor
    feature_vector: torch.Tensor
    categorical_features: Dict[str, torch.Tensor]
    entity_embeddings: Dict[str, torch.Tensor]  # New: embeddings for different entity types
    relationship_matrix: torch.Tensor  # New: relationship adjacency matrix
    metadata: Dict[str, Any]


@dataclass 
class JSONTensorizedOutput:
    """Container for fully processed JSON to tensor conversion"""
    paper_features: torch.Tensor
    author_features: torch.Tensor  
    community_features: torch.Tensor
    relationship_matrix: torch.Tensor
    global_features: torch.Tensor
    metadata: Dict[str, Any]


class JSONToTensorConverter:
    """Converts JSON LLM output to tensors for A2C model"""

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

    def process_json_response(self, json_response: str, response_type: str = "retrieval") -> JSONTensorizedOutput:
        """
        Main method to convert JSON response to tensors
        
        Args:
            json_response: JSON string from LLM
            response_type: "retrieval" or "community_detection"
            
        Returns:
            JSONTensorizedOutput with all tensor features
        """
        try:
            data = json.loads(json_response)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response: {e}")
        
        if response_type == "retrieval":
            return self._process_retrieval_response(data)
        elif response_type == "community_detection":
            return self._process_community_response(data)
        else:
            raise ValueError(f"Unsupported response type: {response_type}")

    def _process_retrieval_response(self, data: Dict) -> JSONTensorizedOutput:
        """Process paper retrieval JSON response"""
        
        entities = data.get("entities", {})
        
        paper_features = self._extract_paper_features(entities.get("papers", []))
         
        author_features = self._extract_author_features(entities.get("authors", []))
        
        relationships = data.get("relationships", {})
        relationship_matrix = self._build_relationship_matrix(entities, relationships)
        
        global_features = self._extract_global_features(data)
        
        # Community features (empty for retrieval)
        community_features = torch.zeros(1, 10)  # Placeholder
        
        return JSONTensorizedOutput(
            paper_features=paper_features,
            author_features=author_features,
            community_features=community_features,
            relationship_matrix=relationship_matrix,
            global_features=global_features,
            metadata={
                "response_type": "retrieval",
                "total_papers": len(entities.get("papers", [])),
                "total_authors": len(entities.get("authors", [])),
                "confidence_level": data.get("query_analysis", {}).get("confidence_level", "unknown")
            }
        )

    def _process_community_response(self, data: Dict) -> JSONTensorizedOutput:
        """Process community detection JSON response"""
        
        communities = data.get("communities", [])
        
        # Extract community features
        community_features = self._extract_community_features(communities)
        
        # Extract entity features from communities
        all_papers = []
        all_authors = []
        
        for community in communities:
            community_entities = community.get("entities", {})
            all_papers.extend(community_entities.get("core_papers", []))
            all_authors.extend(community_entities.get("key_authors", []))
        
        paper_features = self._extract_paper_features(all_papers)
        author_features = self._extract_author_features(all_authors)
        
        # Build inter-community relationship matrix
        relationship_matrix = self._build_community_relationship_matrix(communities)
        
        # Global features
        global_features = self._extract_community_global_features(data)
        
        return JSONTensorizedOutput(
            paper_features=paper_features,
            author_features=author_features,
            community_features=community_features,
            relationship_matrix=relationship_matrix,
            global_features=global_features,
            metadata={
                "response_type": "community_detection",
                "total_communities": len(communities),
                "detection_algorithm": data.get("detection_metadata", {}).get("algorithm_used", "unknown")
            }
        )

    def _extract_paper_features(self, papers: List[Dict]) -> torch.Tensor:
        """Extract tensor features from papers"""
        if not papers:
            return torch.zeros(1, 15)  # Return empty tensor with fixed size
        
        features = []
        for paper in papers:
            paper_features = [
                float(paper.get("citation_count", 0)),
                float(paper.get("relevance_score", 0.0)),
                len(paper.get("authors", [])),
                len(paper.get("keywords", [])),
                self._year_to_recency(paper.get("publication_date", "2020-01-01")),
                float(paper.get("venue", {}).get("impact_factor", 0.0) or 0.0),
                1.0 if paper.get("venue", {}).get("type") == "journal" else 0.0,
                1.0 if paper.get("venue", {}).get("type") == "conference" else 0.0,
            ]
            
            # Text features from title and abstract
            text_content = f"{paper.get('title', '')} {paper.get('abstract', '')}"
            text_features = self._extract_numerical_features_from_text(text_content)
            paper_features.extend(text_features[:7])  # Take first 7 text features
            
            features.append(paper_features)
        
        return torch.tensor(features, dtype=torch.float32)

    def _extract_author_features(self, authors: List[Dict]) -> torch.Tensor:
        """Extract tensor features from authors"""
        if not authors:
            return torch.zeros(1, 10)
        
        features = []
        for author in authors:
            author_features = [
                float(author.get("h_index", 0)),
                float(author.get("total_publications", 0)),
                float(author.get("total_citations", 0)),
                float(author.get("relevance_score", 0.0)),
                len(author.get("affiliations", [])),
                len(author.get("research_areas", [])),
            ]
            
            # Add categorical features
            author_features.extend([
                1.0 if any("professor" in aff.get("position", "").lower() 
                          for aff in author.get("affiliations", [])) else 0.0,
                1.0 if any("university" in aff.get("institution", "").lower() 
                          for aff in author.get("affiliations", [])) else 0.0,
                float(len(author.get("research_areas", [])) > 3),  # Multi-disciplinary
                float(author.get("community_contribution_score", 0.0)) if "community_contribution_score" in author else 0.0
            ])
            
            features.append(author_features)
        
        return torch.tensor(features, dtype=torch.float32)

    def _extract_community_features(self, communities: List[Dict]) -> torch.Tensor:
        """Extract tensor features from communities"""
        if not communities:
            return torch.zeros(1, 20)
        
        features = []
        for community in communities:
            quality_metrics = community.get("quality_metrics", {})
            impact_metrics = community.get("impact_metrics", {})
            diversity_metrics = community.get("diversity_metrics", {})
            
            community_features = [
                float(community.get("relevance_score", 0.0)),
                float(quality_metrics.get("modularity", 0.0)),
                float(quality_metrics.get("conductance", 0.0)),
                float(quality_metrics.get("internal_density", 0.0)),
                float(quality_metrics.get("cohesion_score", 0.0)),
                float(impact_metrics.get("total_citations", 0)),
                float(impact_metrics.get("average_h_index", 0.0)),
                float(impact_metrics.get("recent_publication_rate", 0.0)),
                float(impact_metrics.get("emerging_trend_score", 0.0)),
                float(diversity_metrics.get("geographical_diversity", 0.0)),
                float(diversity_metrics.get("institutional_diversity", 0.0)),
                float(diversity_metrics.get("topical_diversity", 0.0)),
                len(community.get("entities", {}).get("core_papers", [])),
                len(community.get("entities", {}).get("key_authors", [])),
                len(community.get("entities", {}).get("institutions", [])),
            ]
            
            # Temporal features
            temporal_profile = community.get("temporal_profile", {})
            community_features.extend([
                self._year_to_recency(str(temporal_profile.get("formation_year", 2020))),
                self._year_to_recency(str(temporal_profile.get("peak_activity_year", 2020))),
                1.0 if temporal_profile.get("activity_trend") == "emerging" else 0.0,
                1.0 if temporal_profile.get("activity_trend") == "stable" else 0.0,
                1.0 if temporal_profile.get("activity_trend") == "declining" else 0.0,
            ])
            
            features.append(community_features)
        
        return torch.tensor(features, dtype=torch.float32)

    def _build_relationship_matrix(self, entities: Dict, relationships: Dict) -> torch.Tensor:
        """Build relationship adjacency matrix from entities and relationships"""
        
        # Get all entity IDs
        all_entities = []
        for entity_type, entity_list in entities.items():
            if isinstance(entity_list, list):
                all_entities.extend([e.get("paper_id") or e.get("author_id") or e.get("entity_id", f"{entity_type}_{i}") 
                                   for i, e in enumerate(entity_list)])
        
        n_entities = len(all_entities)
        if n_entities == 0:
            return torch.zeros(1, 1)
        
        # Create entity to index mapping
        entity_to_idx = {entity_id: i for i, entity_id in enumerate(all_entities)}
        
        # Initialize adjacency matrix
        adj_matrix = torch.zeros(n_entities, n_entities)
        
        # Process different relationship types
        for rel_type, rel_list in relationships.items():
            if not isinstance(rel_list, list):
                continue
                
            for rel in rel_list:
                if rel_type == "author_collaborations":
                    id1 = rel.get("author_1_id")
                    id2 = rel.get("author_2_id")
                    weight = rel.get("collaboration_strength", 1.0)
                elif rel_type == "paper_citations":
                    id1 = rel.get("citing_paper_id")
                    id2 = rel.get("cited_paper_id")
                    weight = 1.0
                elif rel_type == "topical_similarities":
                    id1 = rel.get("entity_1_id")
                    id2 = rel.get("entity_2_id")
                    weight = rel.get("similarity_score", 0.0)
                else:
                    continue
                
                # Add edges if entities exist in our mapping
                if id1 in entity_to_idx and id2 in entity_to_idx:
                    i, j = entity_to_idx[id1], entity_to_idx[id2]
                    adj_matrix[i, j] = weight
                    adj_matrix[j, i] = weight  # Make symmetric
        
        return adj_matrix

    def _build_community_relationship_matrix(self, communities: List[Dict]) -> torch.Tensor:
        """Build inter-community relationship matrix"""
        n_communities = len(communities)
        if n_communities == 0:
            return torch.zeros(1, 1)
        
        adj_matrix = torch.zeros(n_communities, n_communities)
        
        # Create community ID to index mapping
        community_ids = [c.get("community_id", f"comm_{i}") for i, c in enumerate(communities)]
        id_to_idx = {cid: i for i, cid in enumerate(community_ids)}
        
        # Process cross-community bridges
        for i, community in enumerate(communities):
            bridges = community.get("relationships", {}).get("cross_community_bridges", [])
            for bridge in bridges:
                target_id = bridge.get("target_community_id")
                if target_id in id_to_idx:
                    j = id_to_idx[target_id]
                    bridge_strength = bridge.get("bridge_strength", 0.0)
                    adj_matrix[i, j] = bridge_strength
                    adj_matrix[j, i] = bridge_strength
        
        return adj_matrix

    def _extract_global_features(self, data: Dict) -> torch.Tensor:
        """Extract global features from retrieval response"""
        query_analysis = data.get("query_analysis", {})
        metadata = data.get("metadata", {})
        
        features = [
            len(query_analysis.get("processed_keywords", [])),
            1.0 if query_analysis.get("confidence_level") == "high" else 0.0,
            1.0 if query_analysis.get("confidence_level") == "medium" else 0.0,
            1.0 if query_analysis.get("confidence_level") == "low" else 0.0,
            float(metadata.get("total_entities_found", 0)),
            float(metadata.get("search_depth", 0)),
            float(metadata.get("confidence_distribution", {}).get("high_confidence", 0)),
            float(metadata.get("confidence_distribution", {}).get("medium_confidence", 0)),
            float(metadata.get("confidence_distribution", {}).get("low_confidence", 0)),
        ]
        
        return torch.tensor(features, dtype=torch.float32)

    def _extract_community_global_features(self, data: Dict) -> torch.Tensor:
        """Extract global features from community detection response"""
        detection_metadata = data.get("detection_metadata", {})
        global_analysis = data.get("global_analysis", {})
        
        features = [
            float(detection_metadata.get("total_communities_found", 0)),
            float(detection_metadata.get("query_relevance_threshold", 0.0)),
            len(global_analysis.get("research_landscape", {}).get("dominant_themes", [])),
            len(global_analysis.get("research_landscape", {}).get("emerging_areas", [])),
            len(global_analysis.get("research_landscape", {}).get("declining_areas", [])),
            float(global_analysis.get("network_properties", {}).get("total_nodes", 0)),
            float(global_analysis.get("network_properties", {}).get("total_edges", 0)),
            float(global_analysis.get("network_properties", {}).get("average_path_length", 0.0)),
            float(global_analysis.get("network_properties", {}).get("network_diameter", 0.0)),
        ]
        
        return torch.tensor(features, dtype=torch.float32)

    def extract_community_scores(self, json_response: str) -> List[CommunityScore]:
        """Extract community scores for reward function (updated for new JSON format)"""
        try:
            data = json.loads(json_response)
        except json.JSONDecodeError:
            return []
        
        communities = data.get("communities", [])
        scores = []
        
        for community in communities:
            # Extract scores from the new JSON structure
            relevance = float(community.get("relevance_score", 0.0))
            
            # Calculate quality from quality_metrics
            quality_metrics = community.get("quality_metrics", {})
            quality = np.mean([
                float(quality_metrics.get("modularity", 0.0)),
                float(quality_metrics.get("cohesion_score", 0.0)),
                float(quality_metrics.get("internal_density", 0.0))
            ])
            
            # Calculate diversity from diversity_metrics  
            diversity_metrics = community.get("diversity_metrics", {})
            diversity = np.mean([
                float(diversity_metrics.get("geographical_diversity", 0.0)),
                float(diversity_metrics.get("institutional_diversity", 0.0)),
                float(diversity_metrics.get("topical_diversity", 0.0))
            ])
            
            scores.append(CommunityScore(
                community_id=community.get("community_id", f"community_{len(scores)}"),
                relevance=relevance,
                quality=quality,
                diversity=diversity,
                meta={
                    "impact_metrics": community.get("impact_metrics", {}),
                    "temporal_profile": community.get("temporal_profile", {}),
                    "entity_counts": {
                        "papers": len(community.get("entities", {}).get("core_papers", [])),
                        "authors": len(community.get("entities", {}).get("key_authors", [])),
                        "institutions": len(community.get("entities", {}).get("institutions", []))
                    }
                }
            ))
        
        return scores

    # Helper methods
    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for tokenization"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', ' ', text)
        return text.strip()

    def _extract_numerical_features_from_text(self, text: str) -> List[float]:
        """Extract basic numerical features from text"""
        if not text:
            return [0.0] * 7
        
        features = [
            float(len(text)),
            float(len(text.split())),
            float(len(text.split('.'))),
            float(text.lower().count('research')),
            float(text.lower().count('method')),
            float(text.lower().count('result')),
            float(text.lower().count('conclusion'))
        ]
        return features

    def _year_to_recency(self, date_string: str) -> float:
        """Convert year to recency score (0-1, where 1 is most recent)"""
        try:
            if isinstance(date_string, str):
                year = int(date_string.split('-')[0])
            else:
                year = int(date_string)
            
            current_year = 2024
            max_years_back = 10
            recency = max(0, (current_year - year) / max_years_back)
            return 1.0 - min(1.0, recency)
        except (ValueError, IndexError):
            return 0.5  # Default for invalid dates