import pydantic 
import datetime

class Author(pydantic.BaseModel):
    author_id : str 
    name: str 
    email : str
    affiliation: str
    orcid: str 

class Publication(pydantic.BaseModel):
    paper_id : str
    title : str 
    authors : list[Author]
    date : datetime.date
    journal : str 
    doi: str 

class Cluster(pydantic.BaseModel):
    cluster_id : str 
    name : str 
    description : str
    publications : list[Publication]
    authors : list[Author]
    created_at : datetime.datetime = datetime.datetime.now()
    updated_at : datetime.datetime = datetime.datetime.now()
    
    