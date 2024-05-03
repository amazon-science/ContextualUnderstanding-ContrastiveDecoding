#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate llm

# run experiment with vanilla decoding (without context)
CUDA_VISIBLE_DEVICES=0 python src/contrastive_decoding/run_qa_prompt.py \
 --model_name /home/ec2-user/data/Llama-7b-hf \
 --input_file ./data/popQA.tsv \
 --eval_method vanilla \
 --n_examples 5 \
 --bf16 \
 --alias 'popqa'

# run experiment with regular decoding (with context)
CUDA_VISIBLE_DEVICES=0 python src/contrastive_decoding/run_qa_prompt.py \
 --model_name /home/ec2-user/data/Llama-7b-hf \
 --input_file ./data/popQA.tsv \
 --eval_method contriever \
 --n_examples 5 \
 --ret_path ./data/retrieval/popqa_contriever_results.jsonl \
 --bf16 \
 --alias 'popqa'

# run experiment with CAD decoding (with context)
CUDA_VISIBLE_DEVICES=0 python src/contrastive_decoding/run_qa_prompt.py \
 --model_name /home/ec2-user/data/Llama-7b-hf \
 --input_file ./data/popQA.tsv \
 --eval_method CAD \
 --n_examples 5 \
 --ret_path ./data/retrieval/popqa_contriever_results.jsonl \
 --bf16 \
 --alpha 0.5 \
 --alias 'popqa-alpha-0.5'

# run experiment with our contrastive decoding (with retrieved context)
CUDA_VISIBLE_DEVICES=0 python src/contrastive_decoding/run_qa_prompt.py \
 --model_name /home/ec2-user/data/Llama-7b-hf \
 --input_file ./data/popQA.tsv \
 --eval_method CD \
 --n_examples 5 \
 --ret_path ./data/retrieval/popqa_contriever_results.jsonl \
 --bf16 \
 --alpha 0.5 \
 --alias 'popqa-alpha-0.5'
