import numpy as np

import json

import logging
logger = logging.getLogger(__name__)
 

class HyDE:
    def __init__(self, promptor, generator, encoder, searcher, corpus):
        self.promptor = promptor
        self.generator = generator
        self.encoder = encoder
        self.searcher = searcher
        self.corpus = corpus
    
    def prompt(self, query):
        return self.promptor.build_prompt(query)

    def generate(self, query):
        prompt = self.promptor.build_prompt(query)
        hypothesis_documents = self.generator.generate(prompt)
        return hypothesis_documents
    
    def encode(self, query, hypothesis_documents):
        all_emb_c = []
        for c in [query] + hypothesis_documents:
            c_emb = self.encoder.encode(str(c))
            all_emb_c.append(np.array(c_emb))
        all_emb_c = np.array(all_emb_c)
        avg_emb_c = np.mean(all_emb_c, axis=0)
        hyde_vector = avg_emb_c.reshape((1, len(avg_emb_c)))
        return hyde_vector
    
    def search(self, hyde_vector, k=10):
        hits = self.searcher.search(hyde_vector, k=k)
        return hits
    

    def e2e_search(self, query, k=10):
        prompt = self.promptor.build_prompt(query)
        logger.info(f"Context generation prompt: \n{prompt}")
        hypothesis_documents = self.generator.generate(prompt, num_outputs = 8)
        logger.info("\n\n" + "\n\nHypothesis Context >>>>>> \n\n".join([str(c) for c in hypothesis_documents]))
        hyde_vector = self.encode(query, hypothesis_documents)
        hits = self.searcher.search(hyde_vector, k=k)
        #logger.info(json.loads(self.corpus.doc(hits[0].docid).raw())['contents'])

        for i, hit in enumerate(hits):
            logger.info(f'{i}th HYDE Retrieved Document: {hit.docid}')
            logger.info(json.loads(self.corpus.doc(hit.docid).raw())['contents'])    

        return hits
    

    def e2e_search_flattened(self, query, k=10):
        prompt = self.promptor.build_prompt(query)
        hypothesis_documents = self.generator.generate(prompt, num_outputs = 8)
        all_emb_c = []
        for c in [query] + hypothesis_documents:
            c_emb = self.encoder.encode(str(c))
            all_emb_c.append(np.array(c_emb))
        all_emb_c = np.array(all_emb_c)
        avg_emb_c = np.mean(all_emb_c, axis=0)
        hyde_vector = avg_emb_c.reshape((1, len(avg_emb_c)))
        hits = self.searcher.search(hyde_vector, k = k) 


        for i, hit in enumerate(hits):
            logger.info(f'{i}th HYDE Retrieved Document: {hit.docid}')
            logger.info(json.loads(self.corpus.doc(hit.docid).raw())['contents'])    
            
        return hits
