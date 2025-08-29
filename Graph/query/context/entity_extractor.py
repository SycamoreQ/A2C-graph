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


from vectorstore.base import VectorStoreSearchResult , VectorStoreDocument , BaseVectorStore
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
    

    def map_query_to_entity(
            query: str , 
            text_embedding_vectorstore: BaseVectorStore,
            embedding_vectorstore_ids: str =  EntityVectorStoreKey.ID,
            text_embedder =  EmbeddingModel
    ) -> list[Entity]: 
        
        if query != " " : 
            
            search_result = BaseVectorStore.similarity_search_by_subgraph(
                text = query , 
                text_embedder= lambda t : text_embedder.embed(t)
            )
    


        

    
    def find_entities_using_subgraph(
            entity_name : str ,
            all_entities: list[Entity],
            all_relationships : list[Relationship],
            exclude_entity_names: list[str] | None = None,            
    ) -> list[Entity]:
        if exclude_entity_names is None:
            exclude_entity_names = []
        
        entity_relatiosnships = [
            rel 
            for rel in all_relationships
            if rel.source == entity_name or rel.target == entity_name  
        ]

        source_enttity_subgraph = {rel.source for rel in entity_relatiosnships}
        target_entity_subgraph = {rel.target for rel in entity_relatiosnships}

        #find the similarity between the source and target entities 
        
        
    
        
        
    