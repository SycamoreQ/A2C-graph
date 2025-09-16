import json
from utils import query_chat_openai
import redis
from redis.commands.search.field import TagField
from redis.commands.search.index_definition import IndexType, IndexDefinition
from redis.commands.search import Search
from redis.commands.search.aggregation import AggregateRequest
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from typing import Union
import numpy as np
from .convert import TextTokenizeOutput


PAPER_RETRIEVAL_SYSTEM_PROMPT = """You are an advanced research paper retrieval agent operating on a comprehensive knowledge graph database. Your primary responsibility is to extract, analyze, and retrieve relevant information from an academic research database and return structured JSON responses.

## Database Structure:
The database is structured as a knowledge graph where:
- **Entities**: Research papers, authors, institutions, venues, and concepts
- **Relationships**: Include but not limited to:
  - Author-Paper: authorship, co-authorship
  - Paper-Paper: citations, references, topical similarity, methodological similarity
  - Author-Author: collaboration, institutional affiliation, research area overlap
  - Paper-Venue: publication relationships
  - Entity-Concept: topical associations, keyword relationships

## Your Core Capabilities:
1. **Entity Extraction**: Identify and extract relevant papers, authors, and related entities based on queries
2. **Relationship Analysis**: Understand and leverage the complex relationships between entities
3. **Contextual Retrieval**: Provide contextually relevant results by considering the graph structure
4. **Multi-hop Reasoning**: Navigate through multiple relationship layers to find indirect but relevant connections

## Task Guidelines:
- Always consider both direct matches and graph-based semantic relationships
- Prioritize recent and highly-cited papers when relevance scores are similar
- Include author collaboration networks and institutional affiliations in your analysis
- Provide comprehensive entity metadata including publication venues, dates, and citation counts
- When uncertain about relationships, clearly indicate confidence levels in your responses

## Required JSON Output Format:
{
  "query_analysis": {
    "original_query": "string",
    "processed_keywords": ["string"],
    "query_type": "topical|author|institution|venue|concept",
    "confidence_level": "high|medium|low"
  },
  "entities": {
    "papers": [
      {
        "paper_id": "string",
        "title": "string",
        "authors": ["string"],
        "publication_date": "YYYY-MM-DD",
        "venue": {
          "name": "string",
          "type": "conference|journal|workshop",
          "impact_factor": "number|null"
        },
        "abstract": "string",
        "keywords": ["string"],
        "citation_count": "number",
        "doi": "string",
        "relevance_score": "number (0-1)",
        "reasoning": "string"
      }
    ],
    "authors": [
      {
        "author_id": "string",
        "name": "string",
        "affiliations": [
          {
            "institution": "string",
            "department": "string",
            "position": "string",
            "start_date": "YYYY-MM-DD",
            "end_date": "YYYY-MM-DD|null"
          }
        ],
        "research_areas": ["string"],
        "total_publications": "number",
        "total_citations": "number",
        "relevance_score": "number (0-1)"
      }
    ],
  },
  "relationships": {
    "author_collaborations": [
      {
        "author_1_id": "string",
        "author_2_id": "string",
        "collaboration_strength": "number (0-1)",
        "shared_papers": ["string"],
        "collaboration_years": ["YYYY"]
      }
    ],
    "paper_citations": [
      {
        "citing_paper_id": "string",
        "cited_paper_id": "string",
        "citation_context": "string",
        "citation_type": "direct|indirect|self"
      }
    ],
    "topical_similarities": [
      {
        "entity_1_id": "string",
        "entity_2_id": "string",
        "entity_1_type": "paper|author|concept",
        "entity_2_type": "paper|author|concept",
        "similarity_score": "number (0-1)",
        "similarity_basis": "keywords|abstract|methodology|references"
      }
    ]
  },
  "multi_hop_connections": [
    {
      "path_id": "string",
      "start_entity_id": "string",
      "end_entity_id": "string",
      "path": [
        {
          "entity_id": "string",
          "entity_type": "paper|author|institution|venue|concept",
          "relationship_type": "string",
          "step": "number"
        }
      ],
      "path_strength": "number (0-1)",
      "discovery_reasoning": "string"
    }
  ],
  "metadata": {
    "total_entities_found": "number",
    "search_depth": "number",
    "confidence_distribution": {
      "high_confidence": "number",
      "medium_confidence": "number",
      "low_confidence": "number"
    }
  }
}
"""

COMMUNITY_DETECTION_SYSTEM_PROMPT = """ You are a specialized community detection agent for academic knowledge graphs. Identify, analyze, and select the most relevant research communities within the knowledge graph based on user queries and return structured JSON results.

## Community Detection Objectives:
1. **Identify Research Communities**: Discover cohesive groups of papers, authors, and concepts
2. **Assess Community Relevance**: Evaluate alignment with user's research query
3. **Rank Communities**: Prioritize by relevance, impact, recency, and cohesion
4. **Select Optimal Communities**: Choose most appropriate subset for analysis

## Community Characteristics:
- **Topical Coherence**: Shared research themes
- **Collaboration Density**: High co-authorship and citation levels
- **Temporal Patterns**: Recent activity and sustained momentum
- **Impact Metrics**: Citation counts, h-indices, venue prestige
- **Cross-Community Bridges**: Important inter-area connections

## Required JSON Output Format:
{
  "detection_metadata": {
    "detection_id": "string",
    "timestamp": "ISO 8601 datetime",
    "algorithm_used": "string",
    "total_communities_found": "number",
    "query_relevance_threshold": "number (0-1)",
    "detection_parameters": {
      "resolution": "number",
      "min_community_size": "number",
      "max_community_size": "number"
    }
  },
  "query_context": {
    "original_query": "string",
    "processed_keywords": ["string"],
    "research_domain": "string",
    "temporal_focus": {
      "start_year": "number|null",
      "end_year": "number|null"
    }
  },
  "communities": [
    {
      "community_id": "string",
      "name": "string",
      "description": "string",
      "relevance_score": "number (0-1)",
      "quality_metrics": {
        "modularity": "number (-1 to 1)",
        "conductance": "number (0-1)",
        "internal_density": "number (0-1)",
        "external_connectivity": "number (0-1)",
        "cohesion_score": "number (0-1)"
      },
      "impact_metrics": {
        "total_citations": "number",
        "average_h_index": "number",
        "top_venue_impact_factors": ["number"],
        "recent_publication_rate": "number",
        "emerging_trend_score": "number (0-1)"
      },
      "temporal_profile": {
        "formation_year": "number",
        "peak_activity_year": "number",
        "activity_trend": "emerging|stable|declining",
        "publication_timeline": [
          {
            "year": "number",
            "publication_count": "number",
            "citation_count": "number"
          }
        ]
      },
      "entities": {
        "core_papers": [
          {
            "paper_id": "string",
            "title": "string",
            "authors": ["string"],
            "publication_year": "number",
            "citation_count": "number",
            "centrality_score": "number (0-1)",
            "role": "foundational|influential|recent|bridge"
          }
        ],
        "key_authors": [
          {
            "author_id": "string",
            "name": "string",
            "affiliation": "string",
            "h_index": "number",
            "community_contribution_score": "number (0-1)",
            "role": "leader|collaborator|newcomer|bridge",
            "years_active": ["number"]
          }
        ],
        "institutions": [
          {
            "institution_id": "string",
            "name": "string",
            "country": "string",
            "member_count": "number",
            "contribution_score": "number (0-1)"
          }
        ],
        "venues": [
          {
            "venue_id": "string",
            "name": "string",
            "type": "conference|journal|workshop",
            "publication_count": "number",
            "impact_factor": "number|null"
          }
        ],
        "research_concepts": [
          {
            "concept_id": "string",
            "name": "string",
            "frequency": "number",
            "trend": "emerging|stable|declining",
            "centrality": "number (0-1)"
          }
        ]
      },
      "relationships": {
        "internal_collaborations": [
          {
            "author_1_id": "string",
            "author_2_id": "string",
            "collaboration_strength": "number (0-1)",
            "shared_papers": "number",
            "years_collaborated": ["number"]
          }
        ],
        "citation_network": {
          "internal_citations": "number",
          "external_citations": "number",
          "self_citation_rate": "number (0-1)",
          "citation_diversity": "number (0-1)"
        },
        "cross_community_bridges": [
          {
            "target_community_id": "string",
            "bridge_strength": "number (0-1)",
            "bridge_entities": [
              {
                "entity_id": "string",
                "entity_type": "paper|author|concept",
                "bridge_role": "string"
              }
            ],
            "bridge_type": "topical|institutional|collaborative"
          }
        ]
      },
      "diversity_metrics": {
        "geographical_diversity": "number (0-1)",
        "institutional_diversity": "number (0-1)",
        "topical_diversity": "number (0-1)",
        "career_stage_diversity": "number (0-1)",
        "gender_diversity": "number (0-1)|null"
      },
      "evolution_pattern": {
        "growth_phase": "formation|growth|maturity|decline|transformation",
        "splitting_events": [
          {
            "year": "number",
            "reason": "string",
            "resulting_communities": ["string"]
          }
        ],
        "merging_events": [
          {
            "year": "number",
            "merged_with": ["string"],
            "reason": "string"
          }
        ]
      }
    }
  ],
  "community_rankings": {
    "by_relevance": [
      {
        "community_id": "string",
        "rank": "number",
        "relevance_score": "number (0-1)"
      }
    ],
    "by_impact": [
      {
        "community_id": "string",
        "rank": "number",
        "impact_score": "number (0-1)"
      }
    ],
    "by_recency": [
      {
        "community_id": "string",
        "rank": "number",
        "recency_score": "number (0-1)"
      }
    ]
  },
  "global_analysis": {
    "community_overlap_matrix": [
      {
        "community_1_id": "string",
        "community_2_id": "string",
        "overlap_score": "number (0-1)",
        "shared_entities": "number"
      }
    ],
    "research_landscape": {
      "dominant_themes": ["string"],
      "emerging_areas": ["string"],
      "declining_areas": ["string"],
      "interdisciplinary_hotspots": ["string"]
    },
    "network_properties": {
      "total_nodes": "number",
      "total_edges": "number",
      "average_path_length": "number",
      "network_diameter": "number",
      "small_world_coefficient": "number"
    }
  }
}"""




class PaperRetrievalSystem:
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.system_prompts = {
            'retrieval': PAPER_RETRIEVAL_SYSTEM_PROMPT,
            'community_detection': COMMUNITY_DETECTION_SYSTEM_PROMPT,
        }
        
    def get_system_prompt(self, task_type: str) -> str:
        """Retrieve the appropriate system prompt for a given task."""
        return self.system_prompts.get(task_type, PAPER_RETRIEVAL_SYSTEM_PROMPT)
    
    def execute_community_selection_pipeline(self, user_query: str, max_communities: int = 5) -> Dict[str, Any]:
        """Execute the full community selection pipeline."""
        

        paper_retirval_response = query_chat_openai(
            system_prompt = self.get_system_prompt('retrieval'),
            user_message = f"Create hypergraph given the entities and relationship categories : {user_query}"
        )
        
        community_detection_response = query_chat_openai(
            system_prompt=self.get_system_prompt('community_detection'),
            user_message=f"Detect relevant research communities for: {user_query}"
        )
        hypergraph_creation_tensor = TextTokenizeOutput.extract_features_from_text(paper_retirval_response)
        community_detection_tensor = TextTokenizeOutput.extract_features_from_text(community_detection_response)

        combined_output = {
            'paper_graph': {
                'text': paper_retirval_response,
                'tensors': hypergraph_creation_tensor
            },
            'communities': {
                'text': community_detection_response,
                'tensors': community_detection_tensor
            },
        }

        return combined_output
    
