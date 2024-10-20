import time
import openai 
import math

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)
import logging
from threading import Thread
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
from pydantic import Field, PrivateAttr

import torch 
import hyqe.hyqe.src.utils as utils
 
import os
import yaml

import requests

import tiktoken

from hyqe.hyqe.src.constants import (
    DEFAULT_ENDPOINT_MODEL,
    DEFAULT_ENDPOINT_URL,
    DEFAULT_HUGGINGFACE_MODEL,
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_MAX_NEW_TOKEN,
    DEFAULT_TEMPERATURE,
    DEFAULT_ENDPOINT_SYSTEM_PROMPT,
    DEFAULT_SYSTEM_PROMPT
)
from openai import OpenAI
    
from hyqe.hyqe.src.types import CompletionResponse, LLMMetadata

from hyqe.hyqe.src.promptor import Promptor

from hyqe.hyqe.src.utils import Cache

logger = logging.getLogger(__name__)


class Generator:
    def __init__(self, model_name, topic, n, temperature):
        self.model_name = model_name
        self.topic = topic
        self.n = n
        self.temperature = temperature

        #self.cache = Cache('_'.join([self.model_name.split('/')[-1], self.topic, str(n), str(temperature), 'cache.json']))
        
    def generate(self, prompt):
        if False and prompt in self.cache:
            return self.cache[prompt]
        else:
            resp = self.get_response(prompt)
            #self.cache[prompt] = resp
        return resp


class OpenAIGenerator(Generator):
    def __init__(self, model_name, api_key, n=8, max_tokens=512, temperature=0.7, top_p=1, frequency_penalty=0.0, presence_penalty=0.0, stop=['\n\n\n'], wait_till_success=False):
        super().__init__(model_name, n = n, temperature = temperature)
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop = stop
        self.wait_till_success = wait_till_success
    
    @staticmethod
    def parse_response(response):
        to_return = []
        for _, g in enumerate(response['choices']):
            text = g['text']
            logprob = sum(g['logprobs']['token_logprobs'])
            to_return.append((text, logprob))
        texts = [r[0] for r in sorted(to_return, key=lambda tup: tup[1], reverse=True)]
        return texts

    def get_response(self, prompt):
        
        get_results = False
        while not get_results:
            try:
                result = openai.Completion.create(
                    engine=self.model_name,
                    prompt=prompt,
                    api_key=self.api_key,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    frequency_penalty=self.frequency_penalty,
                    presence_penalty=self.presence_penalty,
                    top_p=self.top_p,
                    n=self.n,
                    stop=self.stop,
                    logprobs=1
                )
                get_results = True
            except Exception as e:
                if self.wait_till_success:
                    time.sleep(1)
                else:
                    raise e
        return self.parse_response(result)
         
        response = openai.client.completions.create(
            prompt=prompt,
            stream=False,
            **all_kwargs,
        )
        text = response.choices[0].text


class CohereGenerator(Generator):
    def __init__(self, model_name, api_key, n=8, max_tokens=512, temperature=0.7, p=1, frequency_penalty=0.0, presence_penalty=0.0, stop=['\n\n\n'], wait_till_success=False):
        super().__init__(model_name)
        self.api_key = api_key
        self.cohere = cohere.Cohere(self.api_key)
        self.n = n
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.p = p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop = stop
        self.wait_till_success = wait_till_success

    
    @staticmethod
    def parse_response(response):
        text = response.generations[0].text
        return text
    
    def generate(self, prompt):
        texts = []
        for _ in range(self.n):
            get_result = False
            while not get_result:
                try:
                    result = self.cohere.generate(
                        prompt=prompt,
                        model=self.model_name,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        frequency_penalty=self.frequency_penalty,
                        presence_penalty=self.presence_penalty,
                        p=self.p,
                        k=0,
                        stop=self.stop,
                    )
                    get_result = True
                except Exception as e:
                    if self.wait_till_success:
                        time.sleep(1)
                    else:
                        raise e
            text = self.parse_response(result)
            texts.append(text)
        return texts



class HuggingfaceGenerator(Generator):
    topic: str = Field(
        default=DEFAULT_HUGGINGFACE_MODEL,
        description=(
            "The tppic"
        ),
    )

    model_name: str = Field(
        default=DEFAULT_HUGGINGFACE_MODEL,
        description=(
            "The model name to use from HuggingFace. "
            "Unused if `model` is passed in directly."
        ),
    )
    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description="The maximum number of tokens available for input.",
        gt=0,
    )
    max_new_tokens: int = Field(
        default=DEFAULT_MAX_NEW_TOKEN,
        description="The maximum number of tokens to generate.",
        gt=0,
    )
    system_prompt: str = Field(
        default="",
        description=(
            "The system prompt, containing any extra instructions or context. "
            "The model card on HuggingFace should specify if this is needed."
        ),
    )
    query_wrapper_prompt: Promptor = Field(
        default="",
        description=(
            "The query wrapper prompt, containing the query placeholder. "
            "The model card on HuggingFace should specify if this is needed. "
            "Should contain a `{}` placeholder."
        ),
    )
    tokenizer_name: str = Field(
        default=DEFAULT_HUGGINGFACE_MODEL,
        description=(
            "The name of the tokenizer to use from HuggingFace. "
            "Unused if `tokenizer` is passed in directly."
        ),
    )
    device_map: str = Field(
        default="auto", description="The device_map to use. Defaults to 'auto'."
    )
    stopping_ids: List[int] = Field(
        default_factory=list,
        description=(
            "The stopping ids to use. "
            "Generation stops when these token IDs are predicted."
        ),
    )
    tokenizer_outputs_to_remove: list = Field(
        default_factory=list,
        description=(
            "The outputs to remove from the tokenizer. "
            "Sometimes huggingface tokenizers return extra inputs that cause errors."
        ),
    )
    tokenizer_kwargs: dict = Field(
        default_factory=dict, description="The kwargs to pass to the tokenizer."
    )
    model_kwargs: dict = Field(
        default_factory=dict,
        description="The kwargs to pass to the model during initialization.",
    )
    generate_kwargs: dict = Field(
        default_factory=dict,
        description="The kwargs to pass to the model during generation.",
    )
 

    _model: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()
    _stopping_criteria: Any = PrivateAttr()

    def __init__(
        self,
        topic: str,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKEN,
        query_wrapper_prompt: str = DEFAULT_SYSTEM_PROMPT,
        tokenizer_name: str = DEFAULT_HUGGINGFACE_MODEL,
        model_name: str = DEFAULT_HUGGINGFACE_MODEL,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        device_map: Optional[str] = "auto",
        stopping_ids: Optional[List[int]] = None,
        tokenizer_kwargs: Optional[dict] = {},
        tokenizer_outputs_to_remove: Optional[list] = [],
        model_kwargs: Optional[dict] = {},
        generate_kwargs: Optional[dict] = {},
        messages_to_prompt: Optional[Callable[[Sequence[str]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        temperature=0.7, 
        n = 1
    ) -> None:
        """Initialize params."""

        super().__init__(
            topic=topic,
            model_name=model_name ,
            temperature=temperature, 
            n=n
        )


        if not os.environ.get('HF_TOKEN', False):
            with open(
                os.path.join(
                    os.path.dirname(
                        os.path.dirname(
                            os.path.dirname(
                                os.path.dirname(__file__)
                            )
                        )
                    ), 'key.yaml'), 'r') as file:
                os.environ['HF_TOKEN'] =  yaml.safe_load(file)['HF_TOKEN']
 
      
        if isinstance(query_wrapper_prompt, str):
            self.query_wrapper_prompt = Promptor(query_wrapper_prompt = query_wrapper_prompt)
        
        self.max_new_tokens = max_new_tokens
        self.tokenizer_outputs_to_remove = tokenizer_outputs_to_remove 
        
        self.tokenizer_name = tokenizer_name
        self.device_map = device_map
        self.stopping_ids = stopping_ids
        self.tokenizer_kwargs = tokenizer_kwargs or {}
        self.model_kwargs = model_kwargs or {}
        self.generate_kwargs = generate_kwargs or {}

        
        self._model = model or AutoModelForCausalLM.from_pretrained(
            model_name, device_map=device_map, **model_kwargs
        )
        

        # check context_window
        config_dict = self._model.config.to_dict()
        model_context_window = int(
            config_dict.get("max_position_embeddings", context_window)
        )
        if model_context_window and model_context_window < context_window:
            logger.warning(
                f"Supplied context_window {context_window} is greater "
                f"than the model's max input size {model_context_window}. "
                "Disable this warning by setting a lower context_window."
            )
            self.context_window = model_context_window
        else:
            self.context_window = context_window

        tokenizer_kwargs = tokenizer_kwargs or {}
        if "max_length" not in tokenizer_kwargs:
            tokenizer_kwargs["max_length"] = context_window

        self._tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            tokenizer_name, **tokenizer_kwargs
        )

        if tokenizer_name != model_name:
            logger.warning(
                f"The model `{model_name}` and tokenizer `{tokenizer_name}` "
                f"are different, please ensure that they are compatible."
            )

        # setup stopping criteria
        stopping_ids_list = stopping_ids or []

        class StopOnTokens(StoppingCriteria):
            def __call__(
                self,
                input_ids: torch.LongTensor,
                scores: torch.FloatTensor,
                **kwargs: Any,
            ) -> bool:
                for stop_id in stopping_ids_list:
                    if input_ids[0][-1] == stop_id:
                        return True
                return False

        self._stopping_criteria = StoppingCriteriaList([StopOnTokens()])
 
        self.messages_to_prompt = utils.messages_to_prompt or self._tokenizer_messages_to_prompt

    @property
    def metadata(self) -> LLMMetadata:
        """LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            max_new_tokens=self.max_new_tokens,
            model_name=self.model_name,
            is_chat_model=self.is_chat_model,
        )

    @classmethod
    def class_name(cls) -> str:
        return "HuggingfaceGenerator"

    def _tokenizer_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Use the tokenizer to convert messages to prompt. Fallback to generic."""
        if hasattr(self._tokenizer, "apply_chat_template"):
            tokens = self._tokenizer.apply_chat_template(messages)
            return self._tokenizer.decode(tokens)

        return utils.messages_to_prompt(messages)

    def prompt_length(self, prompt: str) -> int:
        encoding = tiktoken.encoding_for_model("gpt2")
    
        num_tokens = len(encoding.encode(prompt))

        return num_tokens

    def get_response(
        self, 
        prompt: Union[str, List[str]], 
        formatted: bool = False, 
        **kwargs: Any
    ) -> CompletionResponse:
        """Completion endpoint."""
        full_prompts = prompt if type(prompt) == list else [prompt]
        if not formatted:
            if self.query_wrapper_prompt:
                full_prompts = list(map(self.query_wrapper_prompt.build_prompt, full_prompts))
            if self.system_prompt:
                full_prompts = f"{self.system_prompt} {full_prompts}"

        inputs = self._tokenizer(full_prompts, return_tensors="pt")
        inputs = inputs.to(self._model.device)

        # remove keys from the tokenizer if needed, to avoid HF errors
        for key in self.tokenizer_outputs_to_remove:
            if key in inputs:
                inputs.pop(key, None)
        
        generate_kwargs = {k: v for k, v in self.generate_kwargs.items()}
        if self.n > 1:
            generate_kwargs['do_sample'] = True,
            generate_kwargs['top_k'] = 5
            generate_kwargs['temperature'] = self.temperature
            generate_kwargs['num_return_sequences'] =self.n
        else:
            generate_kwargs['do_sample'] = False,
            generate_kwargs['top_k'] = 5
            generate_kwargs['temperature'] = self.temperature
            generate_kwargs['num_return_sequences'] = 1

        generate_kwargs.update(kwargs)

        tokens = self._model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            stopping_criteria=self._stopping_criteria,
            **generate_kwargs,
        )
  
        completion_tokenss = [tokens_[inputs["input_ids"].size(1):] for tokens_ in tokens[:max(self.n, 1)]]
        
        completions = [self._tokenizer.decode(completion_tokens, skip_special_tokens=True) for completion_tokens in completion_tokenss] 
    
        return [CompletionResponse(text=completion.split('####')[0], raw={"model_output": completion_tokens}) for completion_tokens, completion in zip(completion_tokenss, completions)]
        

class EndPointGenerator(Generator):
    def __init__(
        self, 
        topic: str,
        model_name:str=DEFAULT_ENDPOINT_MODEL,
        api_key:Optional[str] = None,
        url: str = DEFAULT_ENDPOINT_URL,
        system_prompt: Optional[str] = None,
        n = 1,
        temperature = DEFAULT_TEMPERATURE
        ):
        super().__init__(model_name=model_name, topic=topic, n=n, temperature=temperature)

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
        print(f"Load model {self.model_name}")

        self.client = OpenAI(base_url = self.url, api_key=self.api_key)

        if system_prompt is None:
            system_prompt = DEFAULT_ENDPOINT_SYSTEM_PROMPT
        self.system_prompt = system_prompt
 
  
    def get_response(self, text: str, formatted: bool = False):
        headers={
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
            }

        query = {
            "model": self.model_name,
            "temperature": self.temperature,
            "top_p": 1,
            "n": self.n,
            "messages": [
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
                {
                    "role": "user",
                    "content": text,
                }
            ],
            "stream": False
        } 
         
  
        responses = requests.post(f"{self.url}/chat/completions", headers=headers, json=query)
         
        cnt = 0
        if 'choices' not in responses.json():
            if cnt >= 10:
                return None
            #print(query)
            #print(text)
            #print(responses.json())
            resp =  responses.json()
            if 'error' in resp and (
                'retry' in resp['error']['message'] or \
                'try again' in resp['error']['message']
            ):
                time.sleep(20)
                print(f'retry {cnt + 1}')
                responses = requests.post(f"{self.url}/chat/completions", headers=headers, json=query)
                cnt += 1
            else:
                return None
                 
        return list(map(lambda choice: choice['message']['content'].split('####')[0], responses.json()['choices']))

 