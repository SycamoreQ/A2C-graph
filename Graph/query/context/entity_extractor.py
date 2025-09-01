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
from embedder.embedder import TextEmbedder

class EntityVectorStoreKey(str , Enum):
    ID = "id"
    TITLE = "title"
    AUTHOR = "author"
    ATTRIBUTE = "attribute"
    CITATIONS = "citations"

    @staticmethod
    def from_string(value: str) -> "EntityVectorStoreKey":
        """Convert string to EntityVectorStoreKey."""
        if value == "id":
            return EntityVectorStoreKey.ID
        if value == "title":
            return EntityVectorStoreKey.TITLE
        if value == "author":
            return EntityVectorStoreKey.AUTHOR
        if value == "attribute":
            return EntityVectorStoreKey.ATTRIBUTE
        if value == "citations":
            return EntityVectorStoreKey.CITATIONS

        msg = f"Invalid EntityVectorStoreKey: {value}"
        raise ValueError(msg)
    

    def map_query_to_entity(
            query: str , 
            text_embedding_vectorstore: BaseVectorStore,
            embedder: TextEmbedder,
            all_entities: dict[str , Any],
            embedding_vectorstore_key: str = ID,
            include_entity_names: list[str] | None = None,
            exclude_entity_names: list[str] | None = None,
            k: int = 10 

    ) -> list[Entity]: 
        if include_entity_names is None:
            include_entity_names = []

        if exclude_entity_names is None:
            exclude_entity_names = []
        all_entities = list(all_entities.values())

        matched_entities = []
        matched_entities = []

        if query != " " : 
            
            search_result = BaseVectorStore.similarity_search_by_subgraph(
                text = query , 
                text_embedder= lambda t : embedder.encode(t)  
            )

            for results in search_result: 
                if embedding_vectorstore_key == EntityVectorStoreKey.ID and isinstance(
                    results.document.id , str 
                ):
                    matched  = get_entity_by_id(
                        entity_id = results.document.id , 
                        all_entities_dict= all_entities
                    )

                else: 
                    if embedding_vectorstore_key == EntityVectorStoreKey.TITLE or EntityVectorStoreKey.AUTHOR or EntityVectorStoreKey.ATTRIBUTE:
                        matched = get_entity_by_key(
                            entities = all_entities ,
                            key = embedding_vectorstore_key,
                            value = results.document.id
                        )
                        
                    if embedding_vectorstore_key == EntityVectorStoreKey.CITATIONS:
                        matched = get_entities_by_citations(
                            entities = all_entities,
                            min_citations= 1 
                        )


                if matched:
                    matched_entities.append(matched)


        else:
            all_entities.sort(key=lambda x: x.rank if x.rank else 0, reverse=True)
            matched_entities = all_entities[:k]

    # filter out excluded entities
        if exclude_entity_names:
            matched_entities = [
                entity
                for entity in matched_entities
                if entity.title not in exclude_entity_names
            ]

    # add entities in the include_entity list
        included_entities = []
        for entity_name in include_entity_names:
            included_entities.extend(get_entity_by_name(all_entities, entity_name))
        return included_entities + matched_entities


    
        
    
        
        
    