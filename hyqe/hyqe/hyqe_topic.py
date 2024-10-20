from pyserini.search import (
    FaissSearcher, 
    LuceneSearcher,
    LuceneImpactSearcher,
)
from pyserini.index import IndexReader
from pyserini.search.faiss import AutoQueryEncoder
from pyserini.search import get_topics, get_qrels
from tqdm import tqdm
import argparse
import os

from time import sleep
import numpy as np
import shutil
from tqdm import tqdm

import logging
# Set logging level to WARNING
logging.basicConfig(level=logging.WARNING)

import json

import csv


from hyqe.hyqe.src import (
    Promptor, 
    OpenAIGenerator, 
    CohereGenerator, 
    HuggingfaceGenerator, 
    HYQE, 
    HYQE1,  
    EndPointGenerator, 
    EndPointEncoder,
    Encoder
)
from hyqe.hyqe.src.utils import get_doc_content


def write_to_csv(data):
    file_exists = os.path.isfile('log.csv')
     
    with open('log.csv', 'a' if file_exists else 'w', newline='') as csvfile:
        fieldnames = ['datetime', 'model_name', 'method', 'task', 'topic', 
                      'hyqe1-coef', 'hyqe2-coef', 'ent-coef', 'downsample', 'temperature', 'n', 'indexer_name', 
                      'first_n', 'encoder_name', 
                      'indexer_ndcg', 'encoder_ndcg', 'hyqe1_ndcg', 'hyqe2_ndcg', 'hyqe3_ndcg']
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        data_dict = {}
        for field, value in zip(fieldnames, data):
            data_dict[field] = value
        for field in fieldnames[len(data):]:
            data_dict[field] = None

        writer.writerow(data_dict)
 
 
def build_index(topic, indexer_name):
    if topic in ['dl19-passage', 'dl20-passage', 'dl21-passage', 'dl22-passage']:
        corpus = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')

        if indexer_name in ['contriever']:
            query_encoder = AutoQueryEncoder(encoder_dir='facebook/contriever', pooling='mean')
            
            faiss_prebuilt_index_path = os.path.join(
                os.path.dirname(__file__),
                'indexes/contriever_msmarco_index'
            )
           
            searcher = FaissSearcher(faiss_prebuilt_index_path, query_encoder)
            
        elif indexer_name in ['SPLADE++_EnsembleDistil_ONNX', 'hyde_SPLADE++_EnsembleDistil_ONNX']:
            faiss_prebuilt_index_path = None
            lucene_prebuild_index_path="msmarco-v1-passage-splade-pp-ed-text"
            searcher = LuceneImpactSearcher.from_prebuilt_index(
                lucene_prebuild_index_path,
                query_encoder="SpladePlusPlusEnsembleDistil",
                min_idf=0,
                encoder_type="onnx",
            )
            #query_encoder = searcher.object
        elif indexer_name in ['bge-base-en-v1.5']:
            query_encoder = AutoQueryEncoder(encoder_dir='BAAI/bge-base-en-v1.5', pooling='mean')

            faiss_prebuilt_index_path = os.path.join(
                os.path.dirname(__file__),
                'indexes/bge_msmacro_index'
            )
            searcher = FaissSearcher(faiss_prebuilt_index_path, query_encoder)

        if topic == 'dl19-passage':
            topics = get_topics('{}'.format(topic))
            qrels = get_qrels('{}'.format(topic))
        elif topic == 'dl20-passage':
            topics = get_topics('dl20')
            qrels = get_qrels('dl20-passage')
        elif topic == 'dl21-passage':
            topics = get_topics('dl21')
            qrels = get_qrels('dl21-passage')
        elif topic == 'dl22-passage':
            topics = get_topics('dl22')
            qrels = get_qrels('dl22-passage')
    
    elif topic in [
        'beir-v1.0.0-trec-news-test', 
        'beir-v1.0.0-trec-covid-test', 
        'beir-v1.0.0-fever-test',
        'beir-v1.0.0-scidocs-test',
        'beir-v1.0.0-cqadupstack-android-test',
        'beir-v1.0.0-webis-touche2020-test'
        ]:
        topics = get_topics('{}'.format(topic))
        qrels = get_qrels('{}'.format(topic)) 
        

        if 'contriever' in indexer_name:
            query_encoder = AutoQueryEncoder(encoder_dir='facebook/contriever', pooling='mean')
            if topic == 'beir-v1.0.0-trec-covid-test':
                faiss_prebuilt_index_path = "beir-v1.0.0-trec-covid.contriever"
                faiss_prebuilt_index_path = os.path.join(
                os.path.dirname(__file__),
                'indexes/contriever_beir_trec_covid'
                )
                searcher = FaissSearcher(faiss_prebuilt_index_path, query_encoder)
                lucene_prebuilt_index_path="beir-v1.0.0-trec-covid.flat"
            elif topic == 'beir-v1.0.0-trec-news-test':
                faiss_prebuilt_index_path = "beir-v1.0.0-trec-news.contriever"
                lucene_prebuilt_index_path="beir-v1.0.0-trec-news.flat"
                searcher = FaissSearcher.from_prebuilt_index(faiss_prebuilt_index_path, query_encoder)
            elif topic == 'beir-v1.0.0-fever-test':
                faiss_prebuilt_index_path = "beir-v1.0.0-fever.contriever"
                lucene_prebuilt_index_path="beir-v1.0.0-fever.flat"
                searcher = FaissSearcher.from_prebuilt_index(faiss_prebuilt_index_path, query_encoder)
            elif topic == 'beir-v1.0.0-scidocs-test':
                faiss_prebuilt_index_path = "beir-v1.0.0-scidocs.contriever"
                lucene_prebuilt_index_path="beir-v1.0.0-scidocs.flat"
                searcher = FaissSearcher.from_prebuilt_index(faiss_prebuilt_index_path, query_encoder) 
            elif topic == 'beir-v1.0.0-cqadupstack-android-test':
                faiss_prebuilt_index_path = "beir-v1.0.0-cqadupstack-android.contriever"
                lucene_prebuilt_index_path="beir-v1.0.0-cqadupstack-android.flat"
                searcher = FaissSearcher.from_prebuilt_index(faiss_prebuilt_index_path, query_encoder) 
            elif topic == 'beir-v1.0.0-webis-touche2020-test':
                faiss_prebuilt_index_path = "beir-v1.0.0-webis-touche2020.contriever"
                lucene_prebuilt_index_path="beir-v1.0.0-webis-touche2020.flat"
                searcher = FaissSearcher.from_prebuilt_index(faiss_prebuilt_index_path, query_encoder) 
 
        elif indexer_name in ['SPLADE++_EnsembleDistil_ONNX']:
            if topic == 'beir-v1.0.0-trec-news-test':
                lucene_impact_prebuilt_index_path="beir-v1.0.0-trec-news.splade-pp-ed"
                lucene_prebuilt_index_path="beir-v1.0.0-trec-news.flat"
            elif topic == 'beir-v1.0.0-trec-covid-test':
                lucene_impact_prebuilt_index_path="beir-v1.0.0-trec-covid.splade-pp-ed"
                lucene_prebuilt_index_path="beir-v1.0.0-trec-covid.flat"
            elif topic == 'beir-v1.0.0-fever-test':
                lucene_impact_prebuilt_index_path="beir-v1.0.0-fever..splade-pp-ed"
                lucene_prebuilt_index_path="beir-v1.0.0-fever.flat"
            elif topic == 'beir-v1.0.0-scidocs-test':
                lucene_impact_prebuilt_index_path="beir-v1.0.0-scidocs.splade-pp-ed"
                lucene_prebuilt_index_path="beir-v1.0.0-scidocs.flat"
            elif topic == 'beir-v1.0.0-cqadupstack-android-test':
                lucene_impact_prebuilt_index_path="beir-v1.0.0-cqadupstack-android.splade-pp-ed"
                lucene_prebuilt_index_path="beir-v1.0.0-cqadupstack-android.flat"
            elif topic == 'beir-v1.0.0-webis-touche2020-test':
                lucene_impact_prebuilt_index_path="beir-v1.0.0-webis-touche2020.splade-pp-ed"
                lucene_prebuilt_index_path="beir-v1.0.0-webis-touche2020.flat"
            searcher = LuceneImpactSearcher.from_prebuilt_index(
                lucene_impact_prebuilt_index_path,
                query_encoder="SpladePlusPlusEnsembleDistil",
                min_idf=0,
                encoder_type="onnx"
            )
            #query_encoder = searcher.object
        elif indexer_name in ['bge-base-en-v1.5']:
            query_encoder = AutoQueryEncoder(encoder_dir='BAAI/bge-base-en-v1.5', pooling='mean')
            if topic == 'beir-v1.0.0-trec-covid-test':
                faiss_prebuilt_index_path = "beir-v1.0.0-trec-covid.bge-base-en-v1.5"
                lucene_prebuilt_index_path="beir-v1.0.0-trec-covid.flat"
                searcher = FaissSearcher.from_prebuilt_index(faiss_prebuilt_index_path, query_encoder)
            elif topic == 'beir-v1.0.0-trec-news-test':
                faiss_prebuilt_index_path = "beir-v1.0.0-trec-news.bge-base-en-v1.5"
                lucene_prebuilt_index_path="beir-v1.0.0-trec-news.flat"
                searcher = FaissSearcher.from_prebuilt_index(faiss_prebuilt_index_path, query_encoder)
            elif topic == 'beir-v1.0.0-fever-test':
                faiss_prebuilt_index_path = "beir-v1.0.0-fever.bge-base-en-v1.5"
                lucene_prebuilt_index_path="beir-v1.0.0-fever.flat"
                searcher = FaissSearcher.from_prebuilt_index(faiss_prebuilt_index_path, query_encoder)
            elif topic == 'beir-v1.0.0-scidocs-test':
                faiss_prebuilt_index_path = "beir-v1.0.0-scidocs.bge-base-en-v1.5"
                lucene_prebuilt_index_path="beir-v1.0.0-scidocs.flat"
                searcher = FaissSearcher.from_prebuilt_index(faiss_prebuilt_index_path, query_encoder)
            elif topic == 'beir-v1.0.0-cqadupstack-android-test':
                faiss_prebuilt_index_path = "beir-v1.0.0-cqadupstack-android.bge-base-en-v1.5"
                lucene_prebuilt_index_path="beir-v1.0.0-cqadupstack-android.flat"
                searcher = FaissSearcher.from_prebuilt_index(faiss_prebuilt_index_path, query_encoder)
            elif topic == 'beir-v1.0.0-webis-touche2020-test':
                faiss_prebuilt_index_path = "beir-v1.0.0-webis-touche2020.bge-base-en-v1.5"
                lucene_prebuilt_index_path="beir-v1.0.0-webis-touche2020.flat"
                searcher = FaissSearcher.from_prebuilt_index(faiss_prebuilt_index_path, query_encoder)
 
        corpus = LuceneSearcher.from_prebuilt_index(lucene_prebuilt_index_path)
    else:
        raise NotImplementedError

    return searcher, topics, qrels, corpus


def initial_search(topic, doc, indexer_name, searcher, k = 1000, save_path = ''):
    hits_table = {}
    lines = []
    for qid in tqdm(doc):
        if qid in qrels:
            query = doc[qid]['title']
            print("\nQuestion: ", query)
            hits = None
            if isinstance(searcher, FaissSearcher):
                hits = searcher.search(query, k=k, return_vector = True)[1]
            elif isinstance(searcher, LuceneImpactSearcher):
                hits = searcher.search(query, k =k)
            assert hits is not None
            hits_table[qid] = hits
            rank = 0
            for i, hit in enumerate(hits):
                rank += 1
                lines.append(f'{qid} Q0 {hit.docid} {rank} {hit.score} rank\n')
    with open(f'results/{save_path}/{topic}-{indexer_name}-top{k}-trec', 'w')  as f:
        f.writelines(lines)
    return hits_table



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some coefficients.')
    # Add argument for coefficient
    parser.add_argument('--save-path', type = str, help = 'folder to store files', default = '')
    parser.add_argument('--model-name', type = str, help = 'model name', default = 'gpt-3.5-turbo')
    parser.add_argument('--hyqe1-coef', type=float, help='Coefficient value', default = 0.5) 
    parser.add_argument('--ent-coef', type=float, help='Entropy Coefficient value', default = 0.0)
    parser.add_argument('--topic', type=str, help='Topic', default = 'dl19-passage')
    parser.add_argument('--indexer-name', type=str, help='Indexing model', default = 'SPLADE++_EnsembleDistil_ONNX') #bge-base-en-v1.5')#')
    parser.add_argument('--encoder-name', type=str, help='Encoding model', default = 'text-embedding-3-large') #'bge-base-en-v1.5')#'SPLADE++_EnsembleDistil_ONNX')
    parser.add_argument('--task', type=str, help='task template', default = 'web-search')
    parser.add_argument('--method', type=str, help='method', default = 'min')
    parser.add_argument('--first-n', type=int, help='first n reranked', default = 30)
    parser.add_argument('--n', type=int, help='n results', default = 1)
    parser.add_argument('--temperature', type=float, help='temperature', default = 0.1)
    parser.add_argument('--downsample', type=float, help='downsampling ratio', default=1.0)
    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the coefficient value
    save_path = args.save_path
    model_name = args.model_name
    encoder_name = args.encoder_name
    task = args.task
    hyqe1_coef = args.hyqe1_coef 
    ent_coef = args.ent_coef
    topic = args.topic
    method = args.method
    indexer_name = args.indexer_name
    first_n = args.first_n
    n = args.n
    temperature = args.temperature
    downsample = args.downsample

    searcher = None
    corpus = None
    topics = None
    qrel = None
    query_encoder= None

    print(args)
    print(topic)

    
    
    searcher, topics, qrels, corpus = build_index(topic, indexer_name)
    os.makedirs(f'./results/{save_path}', exist_ok=True)
 
    hits_table = initial_search(topic, topics, indexer_name, searcher, k = 100, save_path = '')
    
    promptor = Promptor('hyqe2')
    if topic in ['beir-v1.0.0-scidocs-test']:
        promptor = Promptor('hyqe2_scidocs')
    elif topic in ['beir-v1.0.0-trec-news-test']:
        promptor = Promptor('hyqe2_news')
    elif topic in ['beir-v1.0.0-trec-covid-test']:
        promptor = Promptor('hyqe2_covid')
    elif topic in ['beir-v1.0.0-cqadupstack-android-test']:
        promptor = Promptor('hyqe2_cqadupstack')
    elif topic in ['beir-v1.0.0-webis-touche2020-test']:
        promptor = Promptor('hyqe2_touche2020')

    generator = None
    if model_name == 'mistral-7b-instr':
        selected_model = "mistralai/Mistral-7B-Instruct-v0.2"
        generator = HuggingfaceGenerator(
            topic=topic,
            tokenizer_name=selected_model,
            model_name=selected_model,
            device_map="auto", 
            n = n, 
            temperature = temperature
            # change these settings below depending on your GPU
            #model_kwargs={"torch_dtype": torch.float16, "load_in_8bit": True},
            )
    elif model_name in ["gpt-3.5-turbo", "gpt-4o"]:
        generator = EndPointGenerator(topic=topic, model_name=model_name, n = n, temperature = temperature)
    else:
        raise NotImplementedError(f"{model_name} not found")
    
    encoder= None
    if 'hyd' not in encoder_name:
        searcher = None
        if encoder_name in ['E5-large-v2']:
            encoder = Encoder(encoder_name, AutoQueryEncoder(encoder_dir='intfloat/e5-large-v2', pooling='mean'))#, device = 'cuda:1'))
        elif encoder_name in ['bge-base-en-v1.5']:
            encoder = Encoder(encoder_name, AutoQueryEncoder(encoder_dir='BAAI/bge-base-en-v1.5', pooling='mean'))#, device = 'cuda:1'))
        elif 'contriever' in encoder_name:
            encoder = Encoder(encoder_name, AutoQueryEncoder(encoder_dir='facebook/contriever', pooling='mean'))#, device = 'cuda:1'))
        elif encoder_name in ['text-embedding-3-large']:
            encoder = EndPointEncoder(encoder_name, topic = topic)
        elif encoder_name in ['SFR-Embedding-Mistral']:
            encoder = Encoder(encoder_name, AutoQueryEncoder(encoder_dir='Salesforce/SFR-Embedding-Mistral', pooling='mean', device = 'cuda'), topic = topic)
        elif encoder_name in ['jina-embeddings-v2-base-en']:
            encoder = Encoder(encoder_name, AutoQueryEncoder(encoder_dir='jinaai/jina-embeddings-v2-base-en', pooling='mean'))#,, device = 'cuda:0'), topic = topic)
        elif encoder_name in ['nomic-embed-text-v1.5']:
            encoder = Encoder(encoder_name, AutoQueryEncoder(encoder_dir='nomic-ai/nomic-embed-text-v1.5', pooling='mean'))
    else:
        assert (encoder_name == f'hyd-{indexer_name}' or encoder_name == f'hyde-{indexer_name}') and 'splade' not in indexer_name.lower(), \
            '1. Prebuilt indexes can either be contriever, bge or splade; 2. Encoder must be the same as indexer if using hyde; 3. Splade does not generate embedding thus cannot be used with Hyde'
        encoder = Encoder(encoder_name, searcher.query_encoder)
    
    hyqe = HYQE1(promptor, generator, encoder, searcher, corpus, topic = topic, indexer_name = indexer_name, first_n = first_n)

    lines0 = []
    lines1 = []
     
                    
    for qid in tqdm(topics):
        if qid in qrels:
            query = topics[qid]['title']
            print(f"\n[{save_path}] Question: ", query)

            hits0 = hits_table[qid]
            print(f'[{save_path}] Initial Context:', get_doc_content(corpus.doc(hits0[0].docid)))
             
            if encoder_name != indexer_name:
                if 'hyd' not in encoder_name:
                    hits0 = hyqe.rerank(query=query, hit_cands=hits0, k = 100)
                else:
                    ## Meaning indexer_name and encoder_name are both 'hyde_contriever'
                    ## Then need to do hyde_search since initial_search only uses contriever
                    hits0 = hyqe.hyde_search(query = query, task = task, k = 100, replace_query_emb = 'hyde' in encoder_name)

            print(f'[{save_path}] Rerank Context:', get_doc_content(corpus.doc(hits0[0].docid)))
            rank = 0
            for i, hit in enumerate(hits0[:100]):
                rank += 1
                lines0.append(f'{qid} Q0 {hit.docid} {rank} {hit.score} rank\n')  
                with open(f'results/{save_path}/task-{task}-{topic}-{indexer_name}-{encoder_name}-{model_name}-top100-trec', 'a') as f0, \
                    open(f'results/{save_path}/task-{task}-{topic}-{indexer_name}-{encoder_name}-{model_name}-gen.jsonl', 'a') as fgen0:
                    f0.write(lines0[-1])
                
            
            hits1, scores1, rerank_scores1 = hyqe.variational_rerank(query = query, hit_cands = hits0, hyqe_coef = hyqe1_coef, ent_coef = ent_coef, downsample = downsample, method = method)
            print(f"[{save_path}] HYQE1 Context:", get_doc_content(corpus.doc(hits1[0].docid)))
            rank = 0
            for i, hit in enumerate(hits1[:100]):
                rank += 1
                lines1.append(f'{qid} Q0 {hit.docid} {rank} {scores1[hit.docid]} rank\n')
                with open(f'results/{save_path}/hyqe1-task-{task}-method-{method}-first-{first_n}-hyqe1-coef-{hyqe1_coef}-ent-coef-{ent_coef}-downsample-{downsample}-temperature-{temperature}-n-{n}-{topic}-{indexer_name}-{encoder_name}-{model_name}-top100-trec', 'a') as f1, \
                    open(f'results/{save_path}/hyqe1-task-{task}-method-{method}-first-{first_n}-hyqe1-coef-{hyqe1_coef}-ent-coef-{ent_coef}-downsample-{downsample}-temperature-{temperature}-n-{n}-{topic}-{indexer_name}-{encoder_name}-{model_name}-gen.jsonl', 'a') as fgen1:
                    f1.write(lines1[-1])
          
            
    write_to_csv([save_path, model_name, method, task, topic, 
                      hyqe1_coef, ent_coef, downsample, temperature, n, indexer_name, 
                      first_n, encoder_name])
    
    exit()