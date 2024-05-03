## Enhancing contextual understanding in large language models through contrastive decoding

Large language models (LLMs) tend to inadequately integrate input context during text generation, relying excessively on encoded prior knowledge in model parameters, potentially resulting in generated text with factual inconsistencies or contextually unfaithful content. LLMs utilize two primary knowledge sources: 1) prior (parametric) knowledge from pretraining, and 2) contextual (non-parametric) knowledge from input prompts. The study addresses the open question of how LLMs effectively balance these knowledge sources during the generation process, specifically in the context of open-domain question answering. To address this issue, we introduce a novel approach integrating contrastive decoding with adversarial irrelevant passages as negative samples to enhance robust context grounding during generation. Notably, our method operates at inference time without requiring further training. We conduct comprehensive experiments to demonstrate its applicability and effectiveness, providing empirical evidence showcasing its superiority over existing methodologies.

## Development
First, to create an environment, run the following command using `conda`:
```
conda env create -f environment.yml
```

You will also need to make an [editable install](https://huggingface.co/docs/transformers/installation#editable-install) of Huggingface's `transformers` library since we will need to change the decoding function. 

Once you have installed the library, you can simply need to swap the file `src/contrastive_decoding/lib/transformers/utils.py` in your local copy of the transformers' repository. The path of `utils.py` in the repository should be `src/transformers/generation/`

Then, you can start running experiments:

```
./scripts/run_nq.sh
```

Or alternatively, you can use the following code snippet:

```
CUDA_VISIBLE_DEVICES=0 python src/contrastive_decoding/run_qa_prompt.py \
 --model_name /home/ec2-user/data/Llama-7b-hf \
 --input_file ./data/nq_test.tsv \
 --eval_method CD \
 --n_examples 5 \
 --ret_path ./data/retrieval/nq_contriever_results.jsonl \
 --bf16 \
 --alpha 0.5 \
 --alias 'nq-alpha-0.5'
```
Here are a brief explanation of the args that can be used:
- `--model_name`: this should be the local path or huggingface's model name for the model that you would like to use
- `--input_file`: this should be the file of the QA benchmark in .tsv format
- `--eval_method`: this can be `BM25`, `contriever`, `CD`, `CAD`. `CD` is for our contrastive decoding, and `CAD` stands for context aware decoding, a baseline that we compare with.
- `--n_examples`: the number of examples (shots) for the prompt
- `--ret_path`: this is the path to the retrieval file
- `--alpha`: you should only set this value if you use either `CD` or `CAD`
- `--use_gold_ctx`: set this arg if you would like to use gold context instead of retrieved context
- `--use_fixed_irr`: set this arg if you would like to use one the proivided fixed irrelevant passage
- `--use_random_irr`: set this arg if you would like to randomly select one passage and use it as the irrelevant passage. If both fixed or random are not used, then the default behaviour is to use the most distant passage from the relevent passage as the irrelevant passage. 
- `--alias`: use this arg to set the experiment name that will be used in the saved results (in csv format) 

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the CC-BY-NC-4.0 License.

```
@Inproceedings{Zhao2024,
 author = {Zheng Zhao and Emilio Monti and Jens Lehmann and Haytham Assem},
 title = {Enhancing contextual understanding in large language models through contrastive decoding},
 year = {2024},
 url = {https://www.amazon.science/publications/enhancing-contextual-understanding-in-large-language-models-through-contrastive-decoding},
 booktitle = {NAACL 2024},
}
```