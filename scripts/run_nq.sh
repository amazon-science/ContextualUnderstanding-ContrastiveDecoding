#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate llm


# run experiment with vanilla decoding (without context)
CUDA_VISIBLE_DEVICES=0 python src/contrastive_decoding/run_qa_prompt.py \
 --model_name /home/ec2-user/data/Llama-7b-hf \
 --input_file ./data/nq_test.tsv \
 --eval_method vanilla \
 --n_examples 5 \
 --bf16 \
 --alias 'nq'

# run experiment with regular decoding (with context)
CUDA_VISIBLE_DEVICES=0 python src/contrastive_decoding/run_qa_prompt.py \
 --model_name /home/ec2-user/data/Llama-7b-hf \
 --input_file ./data/nq_test.tsv \
 --eval_method contriever \
 --n_examples 5 \
 --ret_path ./data/retrieval/nq_contriever_results.jsonl \
 --bf16 \
 --alias 'nq'

# run experiment with CAD decoding (with context)
CUDA_VISIBLE_DEVICES=0 python src/contrastive_decoding/run_qa_prompt.py \
 --model_name /home/ec2-user/data/Llama-7b-hf \
 --input_file ./data/nq_test.tsv \
 --eval_method CAD \
 --n_examples 5 \
 --ret_path ./data/retrieval/nq_contriever_results.jsonl \
 --bf16 \
 --alpha 0.5 \
 --alias 'nq-alpha-0.5'

# run experiment with our contrastive decoding (with retrieved context)
CUDA_VISIBLE_DEVICES=0 python src/contrastive_decoding/run_qa_prompt.py \
 --model_name /home/ec2-user/data/Llama-7b-hf \
 --input_file ./data/nq_test.tsv \
 --eval_method CD \
 --n_examples 5 \
 --ret_path ./data/retrieval/nq_contriever_results.jsonl \
 --bf16 \
 --alpha 0.5 \
 --alias 'nq-alpha-0.5'

# run experiment with our contrastive decoding (with gold context)
CUDA_VISIBLE_DEVICES=0 python src/contrastive_decoding/run_qa_prompt.py \
 --model_name /home/ec2-user/data/Llama-7b-hf \
 --input_file ./data/nq_test.tsv \
 --eval_method CD \
 --n_examples 5 \
 --ret_path ./data/retrieval/nq_contriever_results.jsonl \
 --bf16 \
 --use_gold_ctx \
 --alpha 0.5 \
 --alias 'nq-gold-alpha-0.5'