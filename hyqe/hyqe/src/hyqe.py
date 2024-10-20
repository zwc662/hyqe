import numpy as np
import json
import os
from tqdm import tqdm
import re
from functools import reduce
import math

from typing import Any, Dict, List, Callable, Optional, Tuple, Union

from pyserini.search import (
    FaissSearcher, 
    LuceneSearcher,
    LuceneImpactSearcher,
)

from cvxopt import matrix, solvers

import hashlib

from hyqe.hyqe.src.promptor import Promptor
from hyqe.hyqe.src.constants import DEFAULT_TEMPERATURE 
from hyqe.hyqe.src.generator import Generator
from hyqe.hyqe.src import Promptor, HyDE
from hyqe.hyqe.src.utils import get_doc_content, Cache

import logging
logger = logging.getLogger(__name__)


class HYQE:
    def __init__(self, 
                 promptor: Promptor, 
                 generator: Generator, 
                 encoder: Any, 
                 searcher: Any, 
                 corpus: Any,
                 topic: Optional[str] = '',
                 indexer_name: str = 'hyde_contriever'
                 ):
        self.promptor = promptor
        self.generator = generator
        self.encoder = encoder
        self.searcher = searcher
        self.corpus = corpus
        self.indexer_name = indexer_name
        self.query_emb = None
        self.hyqes = {}
        self.hydes = []

        self.topic = topic

        self.hyq_cache = Cache(f"{self.generator.model_name.split('/')[-1]}_{self.topic}_hyq_cache.json")
        if self.generator.temperature != DEFAULT_TEMPERATURE or self.generator.n != 1:
            self.hyq_cache = Cache(f"{self.generator.model_name.split('/')[-1]}_{self.topic}_temperature_{self.generator.temperature}_n_{self.generator.n}_hyq_cache.json")
        print(f"Cache file path: {self.hyq_cache.location}")
        
        self.hyd_cache = None
        #self.encode_cache = Cache(f'{self.topic}_encode_cache.json')

    def hash_order(hit_cands):
        docids = ''.join(map(lambda hit_cand: hit_cand.docid, hit_cands))
        return hashlib.sha256(docids.encode()).hexdigest()
 
    def prompt(self, query: str) -> str:
        return self.promptor.build_prompt(query)
 
    def encode(self, text: str) -> np.ndarray:
        emb = self.encoder.encode(text)
        return np.asarray(emb)
    
        if text in self.encode_cache:
            emb = self.encode_cache[text]
        else:
            if hasattr(self.searcher, 'encoder_type'):
                if self.searcher.encoder_type == 'pytorch':
                    emb = self.encoder.encode(text)
                elif self.searcher.encoder_type == 'onnx':
                    emb = self.encoder.encode_with_onnx(text)
            else:
                emb = self.encoder.encode(text)
            #self.encode_cache[text] = emb.tolist()

        return np.asarray(emb)

    def get_hyqs(self, query) -> List[Any]:
        
        prompt = self.prompt(query)
        if "Content:\n</passage>" in prompt:     
            return ['no content']
        elif prompt in self.hyq_cache:
            hyqs = self.hyq_cache[prompt]
            if len(hyqs) > 0:
                return hyqs
        
          
        lines = []
        resps = self.generator.generate(prompt)
        
        if resps is not None:
            lines = reduce(
                lambda cur_lines, nxt_lines: cur_lines + nxt_lines, 
                map(
                    lambda resp: str(resp).splitlines() if resp is not None else [], 
                    resps
                    )
            )
        else:
            logger.info(f"Generator returned None")
            
        hyqs = []
        for line in lines:
            qs = line.split('?')
            for q in qs:
                if 'Questions:' in q:
                    continue
                q = re.sub(r'^\d+\.|^-\s', '', q.strip().strip('\\').lstrip('\\n').strip('?').lower())
                if len(q) > 5:
                    hyqs.append(q)
            
        self.hyq_cache[prompt] = hyqs
        return hyqs
    
    def get_hyqes(self, docid, hyqs) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
        if docid in self.hyqes:
            return self.hyqes[docid]
        hyqes = []
        hyqe_no_content = None
        for hyq in hyqs:
            if'no content' in hyq.lower() and hyqe_no_content is None:
                hyqe_no_content = self.encode('no content')
            elif hyqe_no_content is not None:
                continue
            else:
                hyqe = self.encode(hyq)
                hyqes.append(hyqe) 
        if len(hyqes) == 0:
            hyqes = [hyqe_no_content]

        hyqes_mean = np.mean(hyqes, axis = 0).flatten()
        hyqes_var = np.var(hyqes, axis = 0)
        self.hyqes[docid] = (hyqes, hyqes_mean, hyqes_var)
        return hyqes, hyqes_mean, hyqes_var
        
    def search(self, vector: Union[List[Any], np.ndarray], k = 10) -> List[Any]:
        input_vector = vector.reshape((1, len(vector)))
        if isinstance(self.searcher, FaissSearcher):
            return self.searcher.search(input_vector, k = k, return_vector = True)[1]
        elif isinstance(self.searcher, LuceneImpactSearcher):
            return self.searcher.search(input_vector, k = k)
      
    
    def get_hyds(self, query, task = 'web search', num_outputs = 8) -> List[Any]:
        prompt = Promptor(task).build_prompt(query)
        logger.info(f"Hypothetical document prompt: \n{prompt}")
        if prompt in self.hyd_cache:
            return self.hyd_cache[prompt]
        else:
            lines = reduce(
                lambda cur_lines, nxt_lines: cur_lines + nxt_lines, 
                map(
                    lambda resp: str(resp).splitlines(), 
                    self.generator.generate(prompt)
                    )
            )
            
            hyds = []
            
            for line in lines:
                hyds += [line.strip()] #[hyd.strip() for hyd in str(hyd_).split("...")]
            logger.info("\n\n" + "\n\nHypothesis Context >>>>>> \n\n" + prompt)
            self.hyd_cache[prompt] = hyds
        return hyds

    def hyde_search(self, query, task = 'web search', k = 10, replace_query_emb = False) -> List[Any]:
        if self.hyd_cache is None:
            self.hyd_cache = Cache(f"{self.generator.model_name.split('/')[-1]}_{self.topic}_hyd_cache.json")
        
        self.query_emb = self.encode(query)
        hyds = self.get_hyds(query, task = task)
        self.hydes = list(map(lambda hyd: self.encode(hyd), hyds))

        all_emb_c = [self.query_emb] + self.hydes
        all_emb_c = np.array(all_emb_c)
        avg_emb_c = np.mean(all_emb_c, axis=0)
        if replace_query_emb:
            self.query_emb = avg_emb_c
        hits = self.search(avg_emb_c, k = k) 
        return hits 
    
    def rerank(self, query, hit_cands, k = None, method: Optional[str] = None) -> List[Any]:
        self.query_emb = self.encode(query)
        query_emb_norm = self.query_emb / np.linalg.norm(self.query_emb, ord = 2)
     
        for hit_cand in hit_cands:
            context = get_doc_content(self.corpus.doc(hit_cand.docid))
            context_emb =  self.encode(context) 
            hit_cand.vectors = context_emb
            if method is None:
                hit_cand.score = np.sum(self.query_emb * context_emb)
            elif method == 'cos' or (hasattr(self.encoder, 'encoder_name') and self.encoder.encoder_name in ['text-embedding-large']):
                context_emb_norm = context_emb / np.linalg.norm(context_emb, ord = 2)
                hit_cand.score = np.sum(query_emb_norm * context_emb_norm)

        hit_cands = sorted(hit_cands, key = lambda hit_cand: -hit_cand.score)
        if k is not None:
            hit_cands = hit_cands[:k]
            
        return hit_cands 
 


class HYQE1(HYQE):
    def __init__(self, 
                 promptor: Promptor, 
                 generator: Generator, 
                 encoder: Any, 
                 searcher: Any, 
                 corpus: Any,
                 topic: Optional[str] = '',
                 indexer_name: str = 'hyde_contriever',
                 first_n: int = 30
                 ):
        super().__init__(promptor, generator, encoder, searcher, corpus, topic, indexer_name)
        self.first_n = first_n
        #self.score_cache = Cache(f'{self.topic}_hyqe1_score_cache.json')

    def get_score(self, hit_cand, ent_coef: Union[float, int], method: str = 'max', downsample: float = 1.0) -> Union[float, int]:
        context = get_doc_content(self.corpus.doc(hit_cand.docid))
        hyqs = set()
        
        if False and hasattr(self.generator, 'context_window'):
            chunk_size = int(self.generator.context_window * 3 / 4)
            cur_size = 0
            prompt_size = self.generator.prompt_length(context)
            if prompt_size > chunk_size:
                '''
                while cur_size < self.generator.prompt_length(context) > chunk_size:
                    context_i = context[cur_size: min(len(context), cur_size + chunk_size)]
                    hyqs.update(self.get_hyqs(context_i))
                    cur_size += int(chunk_size / 2)
                '''
                for _ in range(1 + int(prompt_size / chunk_size)):
                    if chunk_size >= len(context):
                        break
                    context_i = context[chunk_size: min(len(context), cur_size +  int(len(context) / (1 + int(prompt_size / chunk_size))))]
                    cur_size +=  int(len(context) / (1 + int(prompt_size / chunk_size)))
                    hyqs.update(self.get_hyqs(context_i))
                
                hyqs = list(hyqs)
            else:
                hyqs = self.get_hyqs(context)
        else:
            hyqs = self.get_hyqs(context)


        logger.info(hyqs)
        scores = []

        hyqes, hyqe_mean, hyqe_var = None, None, None
         
        for _ in range(10 if downsample < 1.0 else 1):
            hyqs_ = [hyqs[j] for j in np.random.choice(np.arange(len(hyqs)), math.ceil(len(hyqs) * downsample), replace = False).tolist()]
             
            hyqes, hyqe_mean, hyqe_var = self.get_hyqes(hit_cand.docid, hyqs_)
            hyqes = np.asarray(hyqes)
            #score = (hyqe_mean * self.query_emb / np.linalg.norm(hyqe_mean, ord = 2).mean()

            #### Use norm or not ????? #####
            prod = (hyqes @ self.query_emb.reshape(hyqes.shape[1], 1)).flatten() / 1 #np.linalg.norm(self.query_emb, ord = 2)
            hyqes_norm = 1 #np.linalg.norm(hyqes, axis = 1, ord = 2).flatten()
            
            #if method == 'max':
            score_ = (prod / hyqes_norm).max()
            if method == 'mean':
                score_ = (prod / hyqes_norm).mean()
            score_ -= hyqe_var.mean() * ent_coef

            scores.append(score_)

        score = np.mean(scores)
        #assert (type(score) == np.int16 or type(score) == np.float64) and type(context) == str, \
        #    f"[Wront type] score: {type(score)}, context: {type(context)}"
        #self.score_cache[context] = score

        return score, hyqs, hyqes, hyqe_mean, hyqe_var

        
    def variational_rerank(
            self, 
            query: str, 
            hit_cands: Optional[List[Any]] = None, 
            method: str = 'max',
            hyqe_coef: float = 0.4, 
            ent_coef: float = 0.1,
            downsample: float = 1.0
            ) -> List[Any]:  
        
        #if self.query_emb is None:
        self.query_emb = self.encode(query)
        
        hit_scores = {hit_cand.docid: hit_cand.score for hit_cand in hit_cands}
        hit_score_max = max(list(hit_scores.values()))
        hit_score_min = min(list(hit_scores.values()))
        rerank_scores = {hit_cand.docid: 0 for hit_cand in hit_cands}
        self.hyqes = {}
        rank = 0
        for i in range(len(hit_cands)):
            if i >= self.first_n:
                rerank_score = -1e6
                rerank_scores[hit_cand.docid] = rerank_score
                hit_scores[hit_cand.docid] += rerank_score
                continue
            #context_emb = np.asarray(hit_cand.vectors).flatten() #self.encode(context)     
            #context_dir = context_emb / np.linalg.norm(context_emb)
            #score += avg_emb * context_dir * hyqe_coef
            hit_cand = hit_cands[i]
            rerank_score, _, _, _, _ = self.get_score(hit_cands[i], ent_coef, method = method, downsample = downsample)
            if 'SPLADE' in self.indexer_name:
                hit_scores[hit_cand.docid] += rerank_score * hyqe_coef #* (hit_score_max - hit_score_min)
            else:
                hit_scores[hit_cand.docid] += rerank_score * hyqe_coef
            rerank_scores[hit_cand.docid] = rerank_score
        hits = sorted(hit_cands, key = lambda hit_cand: -hit_scores[hit_cand.docid])
                 
        for i, hit in enumerate(hits):
            logger.info(f'{i}th HYQE Rerank Retrieved Document: {hit.docid}')
            logger.info(get_doc_content(self.corpus.doc(hit.docid)))
        
        return hits, hit_scores, rerank_scores
    
