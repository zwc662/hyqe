from abc import ABC, abstractmethod
import numpy as np
from typing import List, Optional, Union, Any, Tuple, Dict
import os
import yaml
import hashlib


from hyqe.hyqe.src.constants import (
    DEFAULT_ENDPOINT_ENCODER,
    DEFAULT_ENDPOINT_URL,
    DEFAULT_HUGGINGFACE_MODEL,
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_NUM_OUTPUTS,
    DEFAULT_TEMPERATURE,
    DEFAULT_ENDPOINT_SYSTEM_PROMPT
)
from hyqe.hyqe.src.utils import Cache, Table

from openai import OpenAI
 

class Encoder(ABC):
    def __init__(self, encoder_name: str, encoder: Any = None, topic: Optional[str] = None):
        self._encoder_name = encoder_name
        self._encoder = encoder
        self._algo = hashlib.sha256()

        self.topic = None
        self._cache = None
        self._table = None

        if topic is not None:
            self.topic = topic
            self._cache = Cache('_'.join([encoder_name.split('/')[-1], self.topic, 'cache.json']))
            self._table = Table('_'.join([encoder_name.split('/')[-1], self.topic, 'cache.parquet']))
            columns = list(self._cache.values())
            self._table.remove_others(columns)

 
    def get_embedding(self, text: str) -> List[float]:
        emb = self._encoder.encode(text)
        if False: #'E5' in self._encoder_name or 'SFR' in self._encoder_name or 'jina' in self._encoder_name or 'nomic' in self._encoder_name: #or 'bge' in self._encoder_name:
            emb = emb / np.linalg.norm(emb, ord = 2)  
        return emb
    
    def encode(self, text: str) -> List[float]:
        embedding = None
        idx = None
        if self.topic is not None:
            if text in self._cache:
                idx = self._cache[text]
                embedding = self._table[idx]
                    
            if idx is None:
                # Update the hash object with the bytes-like object
                self._algo.update(text.encode('utf-8'))
                # Get the hexadecimal representation of the hash
                idx = self._algo.hexdigest()
                self._cache[text] = idx
                
            if embedding is None:
                embedding = self.get_embedding(text)
                self._table[idx] = embedding
        else:
            embedding = self.get_embedding(text)

        embedding = np.asarray(embedding).flatten().tolist()
        return embedding
  

class EndPointEncoder(Encoder):
    def __init__(
        self, 
        encoder_name:str=DEFAULT_ENDPOINT_ENCODER,
        api_key:Optional[str] = None,
        url: str =DEFAULT_ENDPOINT_URL,
        topic: str = ''
        ):
        super().__init__(encoder_name=encoder_name, topic = topic)
        
        if api_key is None:
            with open(
                os.path.join(
                    os.path.dirname(
                        os.path.dirname(
                            os.path.dirname(
                                os.path.dirname(__file__)
                            )
                        )
                    ), 'key.yaml'), 'r') as file:
                api_key= yaml.safe_load(file)['OPENAI_API_KEY']
        self.api_key = api_key
        if url is None:
            url = DEFAULT_ENDPOINT_URL
        self.url = url
        print(f"Load encoder {self._encoder_name}")

        self.client = OpenAI(base_url = self.url, api_key=self.api_key)
    
        '''
        self.topic = topic
         
        self._cache = Cache('_'.join([encoder_name.split('/')[-1], self.topic, 'cache.json']))
        self._table = Table('_'.join([encoder_name.split('/')[-1], self.topic, 'cache.parquet']))
        columns = list(self._cache.values())
        self._table.remove_others(columns)
        '''
        
    def get_embedding(self, text: str):
        embedding = self.client.embeddings.create(input = [text], model=self._encoder_name).data[0].embedding
        return embedding
        
    def encode(self, text: str, cache_only: bool = False, overwrite_cache: bool = False):
        embedding = None
        idx = None
        if text in self._cache or cache_only:
            idx = self._cache[text]
            embedding = self._table[idx]
                   
        if idx is None:
            # Update the hash object with the bytes-like object
            self._algo.update(text.encode('utf-8'))
            # Get the hexadecimal representation of the hash
            idx = self._algo.hexdigest()
            self._cache[text] = idx
             
        if embedding is None or overwrite_cache:
            embedding = self.get_embedding(text)
            self._table[idx] = embedding

        embedding = np.asarray(embedding).flatten().tolist()
        return embedding
  