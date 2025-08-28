from Graph.data.entity import Entity
from typing import List , Dict , Any , cast 
import uuid 
import re 
import pandas as pd 
from collections.abc import Iterable 
import datetime 
import numpy as np 


def get_entity_by_id(entities: dict[str, Entity], value: str) -> Entity | None:
    """Get entity by id."""
    entity = entities.get(value)
    if entity is None and is_valid_uuid(value):
        entity = entities.get(value.replace("-", ""))
    return entity


def get_entity_by_key(
    entities: Iterable[Entity], key: str, value: str | int
) -> Entity | None:
    """Get entity by key."""
    if isinstance(value, str) and is_valid_uuid(value):
        value_no_dashes = value.replace("-", "")
        for entity in entities:
            entity_value = getattr(entity, key)
            if entity_value in (value, value_no_dashes):
                return entity
    else:
        for entity in entities:
            if getattr(entity, key) == value:
                return entity
    return None


def get_entity_by_name(entities: Iterable[Entity], entity_name: str) -> list[Entity]:
    """Get entities by name."""
    return [entity for entity in entities if entity.title == entity_name]


def get_entity_by_attribute(
    entities: Iterable[Entity], attribute_name: str, attribute_value: Any
) -> list[Entity]:
    """Get entities by attribute."""
    return [
        entity
        for entity in entities
        if entity.attributes
        and entity.attributes.get(attribute_name) == attribute_value
    ]

def get_entities_by_citations(
    entities: Iterable[Entity], min_citations: int, key: str | int, value: str | int
) -> List[Entity]:
    """Get important papers with respect to citation density"""
    filtered_entities = get_entity_by_key(entities, key, value)
    if filtered_entities is not None:
        return [
            entity for entity in entities
            if min_citations <= entity.attributes.get("citation_count", 0)
        ]
    return []
        
        
def find_trending_papers(
    entities: Iterable[Entity], 
    days_back: int = 30,
    min_citations_per_day: float = 1.0
) -> List[Entity]:
    """Find papers that are trending based on recent citation velocity."""
    trending = []
    current_date = datetime.now()
    
    for entity in entities:
        if not entity.attributes:
            continue
            
        pub_date_str = entity.attributes.get("publication_date")
        citations = entity.attributes.get("citation_count", 0)
        
        if pub_date_str:
            try:
                pub_date = datetime.strptime(pub_date_str, "%Y-%m-%d")
                days_since_pub = (current_date - pub_date).days
                
                if days_since_pub > 0:
                    citation_velocity = citations / max(days_since_pub, 1)
                    if citation_velocity >= min_citations_per_day:
                        trending.append(entity)
            except ValueError:
                continue
                
    return sorted(trending, key=lambda x: x.attributes.get("citation_count", 0), reverse=True)


def find_seminal_papers(
    entities: Iterable[Entity],
    citation_percentile: float = 90,
    min_age_years: int = 2
) -> List[Entity]:
    """Identify seminal papers based on high citations and age."""
    current_date = datetime.now()
    eligible_papers = []
    
    for entity in entities:
        if not entity.attributes:
            continue
            
        pub_date_str = entity.attributes.get("publication_date")
        citations = entity.attributes.get("citation_count", 0)
        
        if pub_date_str:
            try:
                pub_date = datetime.strptime(pub_date_str, "%Y-%m-%d")
                years_old = (current_date - pub_date).days / 365.25
                
                if years_old >= min_age_years:
                    eligible_papers.append((entity, citations))
            except ValueError:
                continue
    
    if not eligible_papers:
        return []
    
    citations_list = [citations for _, citations in eligible_papers]
    threshold = np.percentile(citations_list, citation_percentile)
    
    return [entity for entity, citations in eligible_papers if citations >= threshold]


def to_entity_dataframe(
    entities: list[Entity],
    include_entity_rank: bool = True,
    rank_description: str = "number of relationships",
) -> pd.DataFrame:
    """Convert a list of entities to a pandas dataframe."""
    if len(entities) == 0:
        return pd.DataFrame()
    header = ["id", "entity", "description"]
    if include_entity_rank:
        header.append(rank_description)
    attribute_cols = (
        list(entities[0].attributes.keys()) if entities[0].attributes else []
    )
    attribute_cols = [col for col in attribute_cols if col not in header]
    header.extend(attribute_cols)

    records = []
    for entity in entities:
        new_record = [
            entity.short_id if entity.short_id else "",
            entity.title,
            entity.description if entity.description else "",
        ]
        if include_entity_rank:
            new_record.append(str(entity.rank))

        for field in attribute_cols:
            field_value = (
                str(entity.attributes.get(field))
                if entity.attributes and entity.attributes.get(field)
                else ""
            )
            new_record.append(field_value)
        records.append(new_record)
    return pd.DataFrame(records, columns=cast("Any", header))


def is_valid_uuid(value: str) -> bool:
    """Determine if a string is a valid UUID."""
    try:
        uuid.UUID(str(value))
    except ValueError:
        return False
    else:
        return True