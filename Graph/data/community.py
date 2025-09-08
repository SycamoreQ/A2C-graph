from dataclasses import dataclass
from typing import Any , Dict , List
import numpy as np 
from data.entity import * 

from .named import Named

@dataclass
class Community(Named):
    "protocol for a community or a set of nodes"
    level: str
    """Community level."""

    type: str 
    """Community type"""

    parent: str
    """Community ID of the parent node of this community."""

    children: list[str]
    """List of community IDs of the child nodes of this community."""

    entities: list[Entity]

    entity_ids: list[str] | None = None
    """List of entity IDs related to the community (optional)."""

    relationship_ids: list[str] | None = None
    """List of relationship IDs related to the community (optional)."""

    text_unit_ids: list[str] | None = None
    """List of text unit IDs related to the community (optional)."""

    covariate_ids: dict[str, list[str]] | None = None
    """Dictionary of different types of covariates related to the community (optional), e.g. claims"""

    attributes: dict[str, Any] | None = None
    """A dictionary of additional attributes associated with the community (optional). To be included in the search prompt."""

    size: int | None = None
    """The size of the community (Amount of text units)."""

    period: str | None = None
    ""

    density: float
    "The density of the community using some metric"

    temporal_evolutions: dict[str , any]
    "unimplemented for now but stores the temporal evolutions of subgraphs based on time slices"



@classmethod
def from_dict(
        cls,
        d: dict[str, Any],
        id_key: str = "id",
        title_key: str = "title",
        short_id_key: str = "human_readable_id",
        level_key: str = "level",
        entities_key: str = "entity_ids",
        relationships_key: str = "relationship_ids",
        text_units_key: str = "text_unit_ids",
        covariates_key: str = "covariate_ids",
        parent_key: str = "parent",
        children_key: str = "children",
        attributes_key: str = "attributes",
        size_key: str = "size",
        period_key: str = "period",
    ) -> "Community":
        """Create a new community from the dict data."""
        return Community(
            id=d[id_key],
            title=d[title_key],
            level=d[level_key],
            parent=d[parent_key],
            children=d[children_key],
            short_id=d.get(short_id_key),
            entity_ids=d.get(entities_key),
            relationship_ids=d.get(relationships_key),
            text_unit_ids=d.get(text_units_key),
            covariate_ids=d.get(covariates_key),
            attributes=d.get(attributes_key),
            size=d.get(size_key),
            period=d.get(period_key),
        )


@dataclass
class CommunityFeatures:
    """Standardized feature representation for RL model training."""
    community_id: str
    feature_vector: np.ndarray  
    metadata: Dict[str, Any]
    confidence_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'community_id': self.community_id,
            'features': self.feature_vector.tolist(),
            'metadata': self.metadata,
            'confidence': self.confidence_score
        }

@dataclass
class RLTrainingInstance:
    """Single training instance for RL model."""
    query_embedding: np.ndarray 
    community_features: List[CommunityFeatures]  
    query_context: Dict[str, Any]  
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'query_embedding': self.query_embedding.tolist(),
            'communities': [c.to_dict() for c in self.community_features],
            'context': self.query_context
        }

# Example feature extraction configuration
FEATURE_SCHEMA = {
    'size_metrics': ['paper_count', 'author_count', 'citation_count'],
    'quality_indicators': ['avg_citation_per_paper', 'avg_h_index', 'venue_impact_score'],
    'temporal_features': ['recency_score', 'publication_velocity', 'temporal_span'],
    'network_properties': ['clustering_coefficient', 'centrality_score', 'connectivity_density'],
    'relevance_scores': ['query_similarity', 'topical_coherence', 'keyword_overlap'],
    'diversity_measures': ['author_diversity', 'methodological_diversity', 'venue_diversity']
}

TOTAL_FEATURE_DIMENSIONS = sum(len(features) for features in FEATURE_SCHEMA.values())