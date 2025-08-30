"""base class for subgraph related context building . This can be applied to local
and global context building 
"""


from Graph.data.entity import Entity
from typing import List , Dict , Any , cast
import uuid 
import re 
import pandas as pd 
import collections.abc 
from collections.abc import Iterable
import datetime 
import numpy as np 
from Graph.data.community import Community
import networkx as nx 
from networkx.algorithms.similarity import graph_edit_distance
from networkx.algorithms.similarity import optimize_edit_paths

from .entity import ( 
    find_seminal_papers,
    find_trending_papers,
    get_entities_by_citations
)



def get_subgraph_by_id(communities: dict[str , Community] , value: str) -> Community | None : 
    """Get subgraph by id"""

    community =  communities.get(value)
    if community is None and is_valid_uuid(value):
        community = communities.get(value.replace("-" , ""))
    return community

def get_subgraph_by_key(
    communities: Iterable[Community], key: str, value: str | int
) -> Community | None:
    """Get entity by key."""
    if isinstance(value, str) and is_valid_uuid(value):
        value_no_dashes = value.replace("-", "")
        for community in communities:
            community_value = getattr(community, key)
            if community_value in (value, value_no_dashes):
                return community
    else:
        for community in communities:
            if getattr(community, key) == value:
                return community
    return None


def get_subgraph_by_name(communities: Iterable[Community], community_name: str) -> list[Community]:
    """Get entities by name."""
    return [community for community in communities if community.title == community_name]


def get_subgraph_by_attribute(
    communities: Iterable[Community], attribute_name: str, attribute_value: Any
) -> list[Community]:
    """Get entities by attribute."""
    return [
        community
        for community in communities
        if community.attributes
        and community.attributes.get(attribute_name) == attribute_value
    ]

def get_subgraph_similarity(communities_source: dict[str , Community] , communities_target: dict[str , Community] , key: str , value : str | int
) -> dict[float , int]:
    
    if communities_source.iter(type) == communities_target.iter(type):
        cs_graph = nx.Graph(communities_source.get(value))
        ct_graph = nx.Graph(communities_target.get(value))

        for i , j in cs_graph , ct_graph:
            v = optimize_edit_paths(cs_graph[i] , ct_graph[j])
            comms_id = get_subgraph_by_id(communities_source , value)
            commt_id = get_subgraph_by_id(communities_target , value)


    data_dict : dict[float , int] = ({"similarity_score" : v  , "community_ids" : comms_id })
    return data_dict

def get_trending_community(
    communities: Iterable[Community], 
    entity_type: str = "paper",
    top_k: int = 5
) -> List[Community]:
    """
    Get trending communities based on citation metrics and recency.
    """
    trending_communities = []
    
    for community in communities:
        if not hasattr(community, 'entities') or not community.entities:
            continue
            
        total_citations = 0
        recent_papers = 0
        entity_count = 0
        current_year = datetime.datetime.now().year
        
        for entity in community.entities:
            if getattr(entity, 'type', '') == entity_type:
                entity_count += 1
                citations = getattr(entity, 'citation_count', 0)
                total_citations += citations
                
                pub_year = getattr(entity, 'publication_year', 0)
                if pub_year >= current_year - 2:
                    recent_papers += 1
        
        if entity_count > 0:
            avg_citations = total_citations / entity_count
            recency_score = recent_papers / entity_count
            trending_score = (0.7 * avg_citations) + (0.3 * recency_score * 100)
            
            trending_communities.append({
                'community': community,
                'trending_score': trending_score,
                'total_citations': total_citations,
                'avg_citations': avg_citations,
                'recent_papers': recent_papers,
                'entity_count': entity_count
            })
    
    trending_communities.sort(key=lambda x: x['trending_score'], reverse=True)
    return [item['community'] for item in trending_communities[:top_k]]


def is_valid_uuid(value: str) -> bool:
    """Determine if a string is a valid UUID."""
    try:
        uuid.UUID(str(value))
    except ValueError:
        return False
    else:
        return True
        
            

    
    