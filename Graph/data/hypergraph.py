from dataclasses import dataclass , field
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import torch 
from torch import tensor
from .named import Named
from .entity import Entity
import numpy as np 

@dataclass 
class HyperedgeType:
    "A Hypergraph type database"
    
    collaborators: str 
    co_authorship: str 
    topical_assosciation: str 
    methodological_group: str
    institutional_affiliation: str 
    citation_cluster:str 
    research_project: str
    cross_disciplinary: str 


@dataclass
class HyperEdge(Named):
    "A hyperedge struct"
    id : str 
    nodes : list[str]
    type : HyperedgeType
    weight : float 
    attributes: dict[str , Any]
    timestamp: Optional[datetime]
    

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "nodes": self.nodes,
            "type": self.type.value,
            "weight": self.weight,
            "attributes": self.attributes,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "confidence": self.confidence
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HyperEdge':
        return cls(
            id=data["id"],
            nodes=data["nodes"],
            type=HyperedgeType(data["type"]),
            weight=data.get("weight", 1.0),
            attributes=data.get("attributes", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else None,
            confidence=data.get("confidence", 1.0)
        )
    

@classmethod
class IncidenceMatrix:
    "The incidence matrix struct . Might use torch tensor for faster computation"
    shape: Tuple[int , int]
    row_indices: List[int]
    column_indices: List[int]
    data: Tuple[torch.tensor]
    node_ids: List[str]
    hyperedge_ids: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shape": self.shape,
            "nnz": len(self.data),
            "row_indices": self.row_indices,
            "col_indices": self.col_indices,
            "data": self.data,
            "node_ids": self.node_ids,
            "hyperedge_ids": self.hyperedge_ids
        }
    

@dataclass 
class HypergraphStatistics:
    """Statistical properties of the hypergraph."""
    total_nodes: int
    total_hyperedges: int
    avg_hyperedge_size: float
    max_hyperedge_size: int
    min_hyperedge_size: int
    avg_node_degree: float 
    size_distribution: Dict[int, int]  # {hyperedge_size: count}
    type_distribution: Dict[str, int]  # {hyperedge_type: count}
    density: float  
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_nodes": self.total_nodes,
            "total_hyperedges": self.total_hyperedges,
            "avg_hyperedge_size": self.avg_hyperedge_size,
            "max_hyperedge_size": self.max_hyperedge_size,
            "min_hyperedge_size": self.min_hyperedge_size,
            "avg_node_degree": self.avg_node_degree,
            "size_distribution": self.size_distribution,
            "type_distribution": self.type_distribution,
            "density": self.density
        }

@dataclass
class HyperGraph: 
    nodes : dict[str , Entity]
    hyperedges: dict[str , HyperEdge]
    metadata: dict[str , Any] = field(default_factory = dict)
    incidence_matrix : Optional[IncidenceMatrix] = None

    def add_node(self , node: Entity):
        self.nodes[node.id] = node

    def add_hyperedge(self , hyper_edge: HyperEdge) -> None:
        self.hyperedges[hyper_edge.id] = hyper_edge
        
    def get_node_hyperedges(self , node_id : str) -> list[HyperEdge]: 
        
        return [he for he in self.hyperedges.values() if node_id in he.nodes]
    

    def get_hyperedge_neighbors(self, node_id: str) -> List[str]:
        """Get all nodes that share at least one hyperedge with the given node."""
        neighbors = set()
        for hyperedge in self.get_node_hyperedges(node_id):
            neighbors.update(hyperedge.nodes)
        neighbors.discard(node_id)  # Remove the node itself
        return list(neighbors)
    
    def compute_statistics(self) -> HypergraphStatistics:
        """Compute and cache hypergraph statistics."""
        if not self.hyperedges:
            return HypergraphStatistics(0, 0, 0, 0, 0, 0, {}, {}, 0.0)
        
        hyperedge_sizes = [len(he.nodes) for he in self.hyperedges.values()]
        node_degrees = {}
        
        # Count node degrees
        for hyperedge in self.hyperedges.values():
            for node_id in hyperedge.nodes:
                node_degrees[node_id] = node_degrees.get(node_id, 0) + 1
        
        # Size distribution
        size_dist = {}
        for size in hyperedge_sizes:
            size_dist[size] = size_dist.get(size, 0) + 1
        
        # Type distribution
        type_dist = {}
        for hyperedge in self.hyperedges.values():
            type_name = hyperedge.type.value
            type_dist[type_name] = type_dist.get(type_name, 0) + 1
        
        # Hypergraph density (edges / possible edges)
        num_nodes = len(self.nodes)
        num_hyperedges = len(self.hyperedges)
        total_possible_connections = sum(hyperedge_sizes)
        max_possible_connections = num_nodes * num_hyperedges
        density = total_possible_connections / max_possible_connections if max_possible_connections > 0 else 0.0
        
        self.statistics = HypergraphStatistics(
            total_nodes=num_nodes,
            total_hyperedges=num_hyperedges,
            avg_hyperedge_size=np.mean(hyperedge_sizes),
            max_hyperedge_size=max(hyperedge_sizes),
            min_hyperedge_size=min(hyperedge_sizes),
            avg_node_degree=np.mean(list(node_degrees.values())) if node_degrees else 0,
            size_distribution=size_dist,
            type_distribution=type_dist,
            density=density
        )
        return self.statistics
    

    def comp_incidence_matrix(self) -> IncidenceMatrix: 
        
        node_ids = list(self.nodes.keys)
        hyperedge_ids = list(self.hyperedges.keys)


        node_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}
        hyperedge_to_idx = {he_id: i for i, he_id in enumerate(hyperedge_ids)}

        row_indices = []
        col_indices = []
        
        for he_id, hyperedge in self.hyperedges.items():
            he_idx = hyperedge_to_idx[he_id]
            for node_id in hyperedge.nodes:
                if node_id in node_to_idx:
                    node_idx = node_to_idx[node_id]
                    row_indices.append(node_idx)
                    col_indices.append(he_idx)
                    data = torch.tensor(hyperedge.weight , dtype = float , device= 'gpu')
        
        self.incidence_matrix = IncidenceMatrix(
            shape=(len(node_ids), len(hyperedge_ids)),
            row_indices=row_indices,
            col_indices=col_indices,
            data=data,
            node_ids=node_ids,
            hyperedge_ids=hyperedge_ids
        )
        return self.incidence_matrix
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert hypergraph to dictionary representation."""
        return {
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "hyperedges": {he_id: he.to_dict() for he_id, he in self.hyperedges.items()},
            "incidence_matrix": self.incidence_matrix.to_dict() if self.incidence_matrix else None,
            "statistics": self.statistics.to_dict() if self.statistics else None,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_entities(cls, entities: List[Entity] ,key: str , value: str ) -> 'HyperGraph':
        """Create a hypergraph from a list of Entity objects."""
        hypergraph = cls()

        for entity in entities: 
            entity_value = getattr(entity , key)
            hypergraph.add_node(entity_value)
        
        return hypergraph


# JSON Schema templates for LLM responses
HYPEREDGE_GENERATION_SCHEMA = {
    "type": "object",
    "properties": {
        "hyperedges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "nodes": {
                        "type": "array", 
                        "items": {"type": "string"}
                    },
                    "type": {
                        "type": "string",
                        "enum": [e.value for e in HyperedgeType]
                    },
                    "weight": {"type": "number", "minimum": 0, "maximum": 1},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "attributes": {"type": "object"}
                },
                "required": ["id", "nodes", "type"]
            }
        },
        "reasoning": {"type": "string"},
        "total_hyperedges": {"type": "integer"}
    },
    "required": ["hyperedges", "total_hyperedges"]
}

HYPERGRAPH_ANALYSIS_SCHEMA = {
    "type": "object", 
    "properties": {
        "communities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "community_id": {"type": "string"},
                    "nodes": {"type": "array", "items": {"type": "string"}},
                    "key_hyperedges": {"type": "array", "items": {"type": "string"}},
                    "community_type": {"type": "string"},
                    "strength": {"type": "number", "minimum": 0, "maximum": 1},
                    "description": {"type": "string"}
                },
                "required": ["community_id", "nodes", "key_hyperedges"]
            }
        },
        "hypergraph_metrics": {
            "type": "object",
            "properties": {
                "modularity": {"type": "number"},
                "clustering_coefficient": {"type": "number"}, 
                "average_path_length": {"type": "number"},
                "density": {"type": "number"}
            }
        }
    },
    "required": ["communities"]
}
        





