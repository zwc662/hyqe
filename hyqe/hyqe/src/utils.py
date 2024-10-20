from abc import ABC, abstractmethod
from pydantic import BaseModel, validator  
import os
import atexit
import time
from tqdm import tqdm

from typing import Any, Awaitable, Callable, List, Optional, Sequence, Dict, Set, Union
from hyqe.hyqe.src.types import MessageRole

import fcntl

import json

import numpy as np

import pyarrow as pa
import pyarrow.parquet as pq

import logging
# Set logging level to WARNING
logging.basicConfig(level=logging.WARNING)


 
 
class Sharable(BaseModel):
    location: str
    lockfile: Optional[str] = None
    checkpoint: Optional[float] = None
    cnt: int = 0


    class Config:
        arbitrary_types_allowed = True
    
    @validator('location')
    def initalizelocation(cls, location: str):  
        ### Location must be inside src/.cache/
        location = os.path.join(
            os.path.dirname(__file__),
            '.cache',
            location
        )
        if not os.path.isdir(os.path.dirname(location)):
            os.makedirs(os.path.dirname(location), exist_ok=True)
        return location
 
    @abstractmethod
    def sync(self): ...
        

class Cache(Sharable):#, metaclass=MonitoredModel):
    data: Dict[Any, Any] = {}
    

    def __init__(self, location: str ='cache.json'):
        super().__init__(location = location)
        self.lockfile = f'{self.location}.lock'  # Lock file name
        self.sync()
        atexit.register(self.exit)

         
    def sync(self):
        # Read the cache file, fetech temporary data, update class data
        temp_data = {}
        
        if os.path.isfile(self.location):
            last_modified = os.path.getmtime(self.location)
            if self.checkpoint is None or last_modified > self.checkpoint: 
                if os.path.isfile(self.location) and os.path.getsize(self.location) > 0:      
                    if True: #with open(self.lockfile, 'w') as lockf:
                        #fcntl.flock(lockf, fcntl.LOCK_EX)
                        try:  
                            with open(self.location, 'r') as fp:
                                temp_data = json.load(fp)
                        finally:
                            # Release file lock
                            #fcntl.flock(lockf, fcntl.LOCK_UN)
                            pass
            self.checkpoint = os.path.getmtime(self.location)
 
        self.data.update(temp_data)
    
    def exit(self):
        with open(self.lockfile, 'w') as lockf:
            self.sync()
            fcntl.flock(lockf, fcntl.LOCK_EX)
            try:
                with open(self.location, 'w') as fp:
                    json.dump(self.data, fp, indent = 6)
                os.chmod(self.location, 0o777)    
            finally:
                # Release file lock
                fcntl.flock(lockf, fcntl.LOCK_UN)
                pass
 
    def __setitem__(self, key: Union[str, int, float], value: Union[str, int, float]):
        #self.sync()

        self.data[key]=value

        '''
        with open(self.lockfile, 'w') as lockf:
            self.sync()
            fcntl.flock(lockf, fcntl.LOCK_EX)
            try:
                with open(self.location, 'w') as fp:
                    json.dump(self.data, fp, indent = 6)
            finally:
                # Release file lock
                fcntl.flock(lockf, fcntl.LOCK_UN)
                self.checkpoint = os.path.getmtime(self.location)
                pass
        '''

    def __getitem__(self, key: Union[str, int, float]) -> Union[str, int, float]:
        return self.data[key]

    def __len__(self) -> int:
        return len(self.data)

    def __contains__(self, key: Union[str, int, float]) -> bool:
        return key in self.data

    def values(self) -> Any:
        return self.data.values()
   

class Table(Sharable):#, metaclass=MonitoredModel):
    table: Optional[pa.Table] = None
    
    def __init__(self, location: str ='table.parquet'):
        super().__init__(location = location)
        self.lockfile = f'{self.location}.lock'  # Lock file name
        self.sync()
        atexit.register(self.exit)

    def exit(self):
        if not self.sync():
            with open(self.lockfile, 'w') as lockf:
                fcntl.flock(lockf, fcntl.LOCK_EX)
                try: 
                    pq.write_table(self.table, self.location)
                    os.chmod(self.location, 0o777) 
                finally:
                    # Release file lock
                    fcntl.flock(lockf, fcntl.LOCK_UN)
                    pass

    def sync(self):
        temp_table = None
        if os.path.isfile(self.location):
            last_modified = os.path.getmtime(self.location)
            if self.checkpoint is None or last_modified > self.checkpoint:   
                if os.path.isfile(self.location) and os.path.getsize(self.location) > 0: 
                    if True: #with open(self.lockfile, 'w') as lockf:
                        #fcntl.flock(lockf, fcntl.LOCK_EX)
                        try: 
                            temp_table= pq.read_table(self.location) 
                        except pa.ArrowInvalid as e:
                            temp_table = None
                        finally:
                            # Release file lock
                            #fcntl.flock(lockf, fcntl.LOCK_UN)'
                            pass
            

        if temp_table is not None:
            self.checkpoint = os.path.getmtime(self.location)
            logging.info("Synchronizing parquet file ...")
            if self.table is None:
                self.table = temp_table
                return False
            cols2add= list(set(temp_table.column_names).difference(set(self.table.column_names)))
            for col_name in tqdm(cols2add):
                #if self.table is not None and field.name not in self.table.column_names:
                self.table = self.table.append_column(col_name, temp_table.column(col_name))
                return True
            return False
        else:
            return False
       
    def __getitem__(self, idx: int):
        if self.table is None or str(idx) not in self.table.column_names:
            logging.info(f"Cannot find idx: {idx}. Return None")
            return None

        idx_column_index = self.table.column_names.index(str(idx))
        idx_column = self.table.column(idx_column_index)
        return idx_column.to_pylist()
        
 
    def __setitem__(self, idx: int, value: Any):
        # Construct a PyArrow table from the new rows
        #

        if self.table is None:
            # Append the new table to the existing table
            self.table = pa.Table.from_arrays([pa.array(value)], names=[str(idx)])
        else:
            if str(idx) in self.table.column_names:
                idx_column_index = self.table.column_names.index(str(idx))
                self.table = self.table.set_column(idx_column_index, str(idx), pa.array(value))
            else:
                self.table = self.table.append_column(str(idx), pa.array(value))
        return    
         
        self.sync()
        
        with open(self.lockfile, 'w') as lockf:
            fcntl.flock(lockf, fcntl.LOCK_EX)
            try: 
                pq.write_table(self.table, self.location)
            finally:
                # Release file lock
                fcntl.flock(lockf, fcntl.LOCK_UN)
                self.checkpoint = os.path.getmtime(self.location)
                pass
        

    def __len__(self):
        if self.table is None:
            return 0
        return len(self.table.column_names)

    def __contains__(self, idx: int) -> bool:
        if self.table is None:
            return False
        return str(idx) in self.table.column_names

    def remove_others(self, columns: List[int]):
        if self.table is None:
            return 
        preserved_columns = list(map(lambda col: str(col), columns))
        schema = self.table.schema
        indices_to_keep = [i for i, field in enumerate(schema) if field.name in preserved_columns]
        # Create a new schema containing only the preserved columns
        self.table = self.table.select(indices_to_keep) 
        pq.write_table(self.table, self.location)
        self.checkpoint = os.path.getmtime(self.location)

def messages_to_prompt(messages: Sequence[Dict[str, str]]) -> str:
    """Convert messages to a prompt string."""
    string_messages = []
    for message in messages:
        role = message.role
        content = message.content
        string_message = f"{role.value}: {content}"

        addtional_kwargs = message.additional_kwargs
        if addtional_kwargs:
            string_message += f"\n{addtional_kwargs}"
        string_messages.append(string_message)

    string_messages.append(f"{MessageRole.ASSISTANT.value}: ")
    return "\n".join(string_messages)


def get_doc_content(doc: Optional[str]):
    if doc is None:
        return ''
    else:
        content = json.loads(doc.raw())

        if "title" in content:
            content = (
                "Title: " + content["title"] + " " + "Content: " + content["text"]
            )
        elif "contents" in content:
            content = content["contents"]
        else:
            content = content["passage"]
        content = " ".join(content.split())
        # hit.score could be of type 'numpy.float32' which is not json serializable. Always explicitly cast it to float.
        return content
                