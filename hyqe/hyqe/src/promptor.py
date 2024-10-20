from typing import Optional, List, Union, Dict
from hyqe.hyqe.src.default_prompts import (
    HYQE1_TMPL_3, HYQE1_TMPL_4, HYQE1_TMPL_6, 
    HYQE1_TMPL_8, HYQE1_TMPL_9, HYQE1_TMPL_10, 
    HYQE1_TMPL_11, HYQE1_TMPL_12,
    REPHRASE_TMPL,
    IRRELEVANT_TMPL,
)
import re

WEB_SEARCH = """Please write a passage to answer the question.
Question: {}
End your passage with ####
Passage:"""


SCIFACT = """Please write a scientific paper passage to support/refute the claim.
Claim: {}
End your passage with ####
Passage:"""


ARGUANA = """Please write a counter argument for the passage.
Passage: {}
End your passage with ####
Counter Argument:"""


TREC_COVID = """Please write a scientific paper passage to answer the question.
Question: {}
End your passage with ####
Passage:"""


FIQA = """Please write a financial article passage to answer the question.
Question: {}
End your passage with ####
Passage:"""


DBPEDIA_ENTITY = """Please write a passage to answer the question.
Question: {}
End your passage with ####
Passage:"""


TREC_NEWS = """Please write a news passage about the topic.
Topic: {}
End your passage with ####
Passage:"""


MR_TYDI = """Please write a passage in {} to answer the question in detail.
Question: {}
End your passage with ####
Passage:"""

TEMPLATE = """
Given a query: {},\n
please write a very short template for the answer.
\n
Use ... as placeholders for the key information.\n 
End your passage with ####\n
"""

 
class Promptor:
    def __init__(self, task: Optional[str] = None, query_wrapper_prompt: Optional[str] = None, language: str = 'en'):
        self.task = task
        self.query_wrapper_prompt = query_wrapper_prompt
        self.language = language
    
    def build_prompt(self, query: Union[str, Dict[str, str]]):
        if self.task == 'template':
            prompt = TEMPLATE.format(query)
        elif self.task in ['web search', 'web-search']:
            prompt = WEB_SEARCH.format(query)
        elif self.task == 'scifact':
            prompt = SCIFACT.format(query)
        elif self.task == 'arguana':
            prompt = ARGUANA.format(query)
        elif self.task == 'trec-covid':
            prompt = TREC_COVID.format(query)
        elif self.task == 'fiqa':
            prompt = FIQA.format(query)
        elif self.task == 'dbpedia-entity':
            prompt = DBPEDIA_ENTITY.format(query)
        elif self.task == 'trec-news':
            prompt = TREC_NEWS.format(query)
        elif self.task == 'mr-tydi':
            prompt = MR_TYDI.format(self.language, query)
        elif self.task == 'hyqe1':
            prompt = HYQE1_TMPL_9.format(query) #HYQE1_TMPL_6.format(query)
        elif self.task == 'hyqe2':
            prompt = HYQE1_TMPL_9.format(query) #HYQE1_TMPL_6.format(query)
        elif self.task == 'hyqe2_news':
            prompt = HYQE1_TMPL_9.format(query)
        elif self.task == 'hyqe2_covid':
            prompt = HYQE1_TMPL_9.format(query)
        elif self.task == 'hyqe2_scidocs':
            prompt = HYQE1_TMPL_10.format(query)
        elif self.task == 'hyqe2_cqadupstack':
            prompt = HYQE1_TMPL_11.format(query)
        elif self.task == 'hyqe2_touche2020':
            query = re.sub(r'(\bContent:)', r'\n\1', query)
            prompt = HYQE1_TMPL_12.format(query)
        elif self.task == "rephrase":
            prompt = REPHRASE_TMPL.format(query)
        elif self.task == 'irrelevant':
            prompt = IRRELEVANT_TMPL.format(query)
        elif self.task is None:
            prompt = query
        else:
            raise ValueError(f"Task '{self.task}'  not supported")
        
        if self.query_wrapper_prompt is not None:
            prompt = self.query_wrapper_prompt.format(prompt)
        return prompt
