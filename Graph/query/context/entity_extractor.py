from Graph.data import entity
from Graph.data import relation 
from Graph.retrieval.entity import (
    get_entity_by_id,
    get_entity_by_key,
    get_entity_by_name,
    get_entity_by_attribute,
    get_entities_by_citations,
    find_seminal_papers,
    find_trending_papers 
)
from enum import Enum

from vectorstore import BaseVectorStore
from typing import List , Dict , Any
from Graph.data.entity import Entity
from Graph.data.relationship import Relationship

class EntityVectorStoreKey(str , Enum):
    ID = "id"
    TITLE = "title"

    @staticmethod
    def from_string(value: str) -> "EntityVectorStoreKey":
        """Convert string to EntityVectorStoreKey."""
        if value == "id":
            return EntityVectorStoreKey.ID
        if value == "title":
            return EntityVectorStoreKey.TITLE

        msg = f"Invalid EntityVectorStoreKey: {value}"
        raise ValueError(msg)

    
    def find_entities_using_subgraph(
            entity_name : str ,
            all_entities: list[Entity],
            all_relationships : list[Relationship],            
    ) -> list[Entity]:
        
    