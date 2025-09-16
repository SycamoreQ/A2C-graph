import torch 
from typing import List , Any , Dict , Tuple , Optional
from dataclasses import dataclass
import logging
import numpy as np
from model.llm import PaperRetrievalSystem


@dataclass
class State: 

    def __init__(self , features: torch.Tensor):
        self.features = features

        
    def parse_paper(self , pipeline: Dict[str , Any]):
        
        if 'paper_graph' in pipeline.keys: 
            pipeline_text = pipeline['paper_graph']['tensors']
    

    
            