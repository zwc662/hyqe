for model_name in  gpt-3.5-turbo mistral-7b-instr gpt-4o; do # Choosing hypothetical query generator
for indexer_name in contriever bge-base-en-v1.5 SPLADE++_EnsembleDistil_ONNX; do # Choosing retriever
for encoder_name in nomic-embed-text-v1.5 contriever E5-large-v2 bge-base-en-v1.5 text-embedding-3-large ; do ## Chossing embedding model
for entcoef in 0.0; do
for task in web-search; do
for method in max mean; do ### Choosing Eq2 or Eq5 as the objective 
for topic in beir-v1.0.0-trec-news-test beir-v1.0.0-trec-covid-test dl19-passage dl20-passage; do ## Choosing datasets 
for first_n in 30 20 10; do ### Select K for C_{q, K}
for entcoef in 0.0; do # Add entropy loss as in Eq.5
for downsample in 1.0; do # Downsample the generated hypothetical queries
for n in 1 ; do # number of outputs (LLM generator API parameter)
for temperature in 1.0; do # temperature  (LLM generator API parameter)


## Set hyperparameter $\lambda$

if [[ $encoder_name == bge-base-en-v1.5 ]]; then
    hyqe1coef=0.03
fi


if [[ $encoder_name == contriever ]]; then
    hyqe1coef=2.0
fi

if [[ $encoder_name == text-embedding-3-large ]]; then
    hyqe1coef=0.3
fi

if [[ $encoder_name == E5-large-v2 ]]; then
    hyqe1coef=0.5
fi


if [[ $encoder_name == SFR-Embedding-Mistral ]]; then
    hyqe1coef=0.5
fi

if [[ $encoder_name == nomic-embed-text-v1.5 ]]; then
    hyqe1coef=0.5
fi

current_date_time=$(date +'%Y%m%d%H%M%S');
mkdir -p results/${current_date_time}
commit_id=$(git rev-parse HEAD)
 

echo ">>>>>>>" >> log.txt
echo "datetime: ${current_date_time}" | tee -a log.txt >> results/${current_date_time}/log.txt
echo "commit: ${commit_id}" | tee -a log.txt >> results/${current_date_time}/log.txt
echo "model_name: ${model_name}" | tee -a log.txt >> results/${current_date_time}/log.txt
echo "method: ${method}" | tee -a log.txt >> results/${current_date_time}/log.txt
echo "task: ${task}" | tee -a log.txt >> results/${current_date_time}/log.txt
echo "topic: ${topic}" | tee -a log.txt >> results/${current_date_time}/log.txt
echo "hyqe1-coef: ${hyqe1coef}" | tee -a log.txt >> results/${current_date_time}/log.txt 
echo "ent-coef: ${entcoef}" | tee -a log.txt >> results/${current_date_time}/log.txt
echo "indexer_name: ${indexer_name}" | tee -a log.txt >> results/${current_date_time}/log.txt
echo "fist_n: ${first_n}" | tee -a log.txt >> results/${current_date_time}/log.txt
echo "encoder_name: ${encoder_name}" | tee -a log.txt >> results/${current_date_time}/log.txt
echo "downsample: ${downsample}" | tee -a log.txt >> results/${current_date_time}/log.txt
echo "temperature: ${temperature}" | tee -a log.txt >> results/${current_date_time}/log.txt
echo "n: ${n}" | tee -a log.txt >> results/${current_date_time}/log.txt


python hyqe_topic.py --save-path ${current_date_time} --first-n ${first_n} --encoder-name ${encoder_name} --model-name ${model_name} --hyqe1-coef ${hyqe1coef} --ent-coef ${entcoef} --topic ${topic} --task ${task} --method ${method} --indexer-name ${indexer_name} --downsample ${downsample} --temperature ${temperature} --n ${n}

echo "<<<<<<<" >> log.txt
echo "datetime: ${current_date_time}" >> log.txt
echo "commit: ${commit_id}" >> log.txt
echo "model_name: ${model_name}" >> log.txt
echo "method: ${method}" >> log.txt
echo "task: ${task}" >> log.txt
echo "topic: ${topic}" >> log.txt
echo "hyqe1-coef: ${hyqe1coef}" >> log.txt 
echo "ent-coef: ${entcoef}" >> log.txt
echo "indexer_name: ${indexer_name}" >> log.txt
echo "fist_n: ${first_n}" >> log.txt
echo "encoder_name: ${encoder_name}" >> log.txt
echo "downsample: ${downsample}" >> log.txt
echo "temperature: ${temperature}" >> log.txt
echo "n: ${n}" >> log.txt


######## Run tests to get ndcg@10 score
for metric in ndcg_cut.10; do

# Test retriever
echo "python -m pyserini.eval.trec_eval -c -l 2 -m ${metric} ${topic} results/${topic}-${indexer_name}-top100-trec" | tee -a log.txt >> results/${current_date_time}/log.txt
python -m pyserini.eval.trec_eval -c -l 2 -m ${metric} ${topic} results/${topic}-${indexer_name}-top100-trec | tee -a log.txt >> results/${current_date_time}/log.txt

# Test embedding model baseline
echo "python -m pyserini.eval.trec_eval -c -l 2 -m ${metric} ${topic} results/${current_date_time}/task-${task}-${topic}-${indexer_name}-${encoder_name}-${model_name}-top100-trec" | tee -a log.txt >> results/${current_date_time}/log.txt
python -m pyserini.eval.trec_eval -c -l 2 -m ${metric} ${topic} results/${current_date_time}/task-${task}-${topic}-${indexer_name}-${encoder_name}-${model_name}-top100-trec | tee -a log.txt >> results/${current_date_time}/log.txt

# Test HyQE
algo=hyqe1
echo "python -m pyserini.eval.trec_eval -c -l 2 -m ${metric} ${topic} results/${current_date_time}/${algo}-task-${task}-method-${method}-first-${first_n}-hyqe1-coef-${hyqe1coef}-hyqe2-coef-${hyqe2coef}-ent-coef-${entcoef}-${topic}-${indexer_name}-${encoder_name}-${model_name}-top100-trec" | tee -a log.txt >>  results/${current_date_time}/log.txt
python -m pyserini.eval.trec_eval -c -l 2 -m ${metric} ${topic} results/${current_date_time}/${algo}-task-${task}-method-${method}-first-${first_n}-hyqe1-coef-${hyqe1coef}-hyqe2-coef-${hyqe2coef}-ent-coef-${entcoef}-${topic}-${indexer_name}-${encoder_name}-${model_name}-top100-trec | tee -a log.txt >> results/${current_date_time}/log.txt
done
done

done
done
done
done
done
done
done
done
done
done
done