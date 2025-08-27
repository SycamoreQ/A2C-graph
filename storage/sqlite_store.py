import sqlite3 
import json
import logging
import re
from collections.abc import Iterator
from datetime import datetime, timezone
from io import BytesIO, StringIO
from typing import Any
from storage.pipeline_storage import PipelineStorage


class SQLStore[PipelineStorage]:
    """A simple SQLite based storage implementation"""
    
    def __init__(self , db)