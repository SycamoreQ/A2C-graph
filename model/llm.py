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


PAPER_RETRIEVAL_SYSTEM_PROMPT = """You are an advanced research paper retrieval agent operating on a comprehensive knowledge graph database. Your primary responsibility is to extract, analyze, and retrieve relevant information from an academic research database.

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

## Output Format:
Structure your responses with clear entity identification, relationship mappings, and relevance scores. Include reasoning for your selections and highlight any important indirect connections discovered through graph traversal."""


HYPERGRAPH_CREATION_SYSTEM_PROMPT = """You are a specialized agent that takes a given set of entities and relationships and create a hypergraph out of both inputs. Given a large entity and relationship triple database , your job is to create a hypergraph that will connect 
two entities or more than two entities with respect to their relationships. The database that will be given to you is a research paper database with Research papers, authors, institutions, venues, and concepts as entities and these following relationships: 
-collaborators : the people who worked on the publication
-co authorships : the authors of the paper
-methodological group : the laboratory the paper was written in
-institutional affiliation : the institute the author or the paper was written in 
-citation clusters : the citations that specific paper has 
-cross disciplinary: the different plethora of subjects that paper contributes to.

## Your core capabilities: 
1. **Entity Extraction: Identify the research papers , authors , institutions , venues based on the query
2. **Relationship Analysis: Understand the complex relationships between the entities
3. **Hypergraph Construction: Create the hypergraph by connecting relationships that overlap two or more entities together , you should classify that as a hyperedge and return the hyperedge and the entities it encompasses

# Task Guidlines:
- Only consider the relationships to be a hyperedge if there is a clear sign of a overlap of any given reason stated above. The hyperedge has to be under those categories or else it is not considered a hyperedge
- There should also be a clear topological understanding of the hypergraph i.e do not make it very sparse and not too dense as well. Filter based on you knowledge

#Output format: 
Structure you responses with clear entity and hyperedge relations. Include the attribute or edge type based on the categories mentioned on top
"""

COMMUNITY_DETECTION_SYSTEM_PROMPT = """You are a specialized community detection agent for academic knowledge graphs. Your role is to identify, analyze, and select the most relevant research communities within the knowledge graph based on user queries and graph structure.

## Community Detection Objectives:
1. **Identify Research Communities**: Discover cohesive groups of papers, authors, and concepts that form meaningful research communities
2. **Assess Community Relevance**: Evaluate how well each community aligns with the user's research query or interest
3. **Rank Communities**: Prioritize communities based on relevance, impact, recency, and internal cohesion
4. **Select Optimal Communities**: Choose the most appropriate subset of communities for detailed analysis

## Community Characteristics to Consider:
- **Topical Coherence**: Papers and authors sharing similar research themes
- **Collaboration Density**: High levels of co-authorship and citation within the community
- **Temporal Patterns**: Recent activity and sustained research momentum
- **Impact Metrics**: Citation counts, h-indices, and venue prestige within communities
- **Cross-Community Bridges**: Important connections between different research areas

## Output Requirements:
Provide the list of communities with proper entities involved to create that community and the relations they have , communities relevance , 
quality and diversity amongst the other communities.
"""

QUERY_REFINEMENT_SYSTEM_PROMPT = """You are a query refinement specialist for academic knowledge graphs. Your role is to analyze user queries and suggest improvements, expansions, or alternative formulations to maximize retrieval effectiveness.

## Query Analysis Process:
1. **Intent Recognition**: Identify the user's underlying research intent and goals
2. **Concept Extraction**: Extract key technical terms, methodologies, and research areas
3. **Ambiguity Detection**: Identify potentially ambiguous terms or concepts
4. **Context Enhancement**: Suggest additional context that could improve results
5. **Alternative Formulations**: Propose different ways to express the same information need

## Refinement Strategies:
- **Term Expansion**: Add synonyms, related terms, and alternative phrasings
- **Specificity Adjustment**: Make queries more specific or more general as appropriate
- **Temporal Constraints**: Suggest time-based filters when relevant
- **Methodological Focus**: Highlight specific approaches or techniques of interest
- **Interdisciplinary Connections**: Identify potential cross-domain relevance
- **Author and Venue Targeting**: Suggest specific researchers or publication venues

## Output Format:
- Original query analysis and interpretation
- Identified potential issues or limitations
- Refined query suggestions with explanations
- Alternative query formulations for different perspectives
- Recommended search strategies and filters"""

class PaperRetrievalSystem:
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.system_prompts = {
            'retrieval': PAPER_RETRIEVAL_SYSTEM_PROMPT,
            'hypergraph_creation': HYPERGRAPH_CREATION_SYSTEM_PROMPT , 
            'community_detection': COMMUNITY_DETECTION_SYSTEM_PROMPT,
            'query_refinement': QUERY_REFINEMENT_SYSTEM_PROMPT
        }
        
    def get_system_prompt(self, task_type: str) -> str:
        """Retrieve the appropriate system prompt for a given task."""
        return self.system_prompts.get(task_type, PAPER_RETRIEVAL_SYSTEM_PROMPT)
    
    def execute_community_selection_pipeline(self, user_query: str, max_communities: int = 5) -> Dict[str, Any]:
        """Execute the full community selection pipeline."""
        
        refined_query_response = query_chat_openai(
            system_prompt=self.get_system_prompt('query_refinement'),
            user_message=f"Analyze and refine this research query: {user_query}"
        )

        hypergraph_creation_response = query_chat_openai(
            system_prompt = self.get_system_prompt('hypergraph_creation'),
            user_message = f"Create hypergraph given the entities and relationship categories : {user_query}"
        )
        
        community_detection_response = query_chat_openai(
            system_prompt=self.get_system_prompt('community_detection'),
            user_message=f"Detect relevant research communities for: {user_query}"
        )

        refined_query_tensor = TextTokenizeOutput.extract_features_from_text(refined_query_response)
        hypergraph_creation_tensor = TextTokenizeOutput.extract_features_from_text(refined_query_response)
        community_detection_tensor = TextTokenizeOutput.extract_features_from_text(community_detection_response)

        combined_output = {
            'refined_query': {
                'text': refined_query_response,
                'tensors': refined_query_tensor
            },
            'hypergraph': {
                'text': hypergraph_creation_response,
                'tensors': hypergraph_creation_tensor
            },
            'communities': {
                'text': community_detection_response,
                'tensors': community_detection_tensor
            },
        }

        return combined_output
    

    
    

        




