#!/usr/bin/python
# -*- coding: UTF-8 -*-

import random
import torch
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import argparse

seed = 2023

torch.backends.cudnn.deterministic = True
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
print('Cuda:', torch.cuda.is_available())
print('Number of GPUs:', torch.cuda.device_count())
print('pwd', os.getcwd())

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM
# from util_clm import convert_model_to_int8_on_gpu

import jsonlines

def load_jsonlines(file):
    with jsonlines.open(file, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst


# completion_template = "Q: {} A:"  # "{}" # "Query: {}\nResult:" # "Q: {} A:" # "{} The answer is"
completion_template = "Answer the following question:\n\n{question}"  # "{}" # "Query: {}\nResult:" # "Q: {} A:" # "{} The answer is"
completion_template_context = "Answer based on context:\n\n{context}\n\n{question}"
genread_template = "Generate a background document from Wikipedia to answer the given question. {}"  # This prompt comes from the GenRead paper

    
def call_model(prompt, prompt_rel, prompt_irr, alpha, model, tokenizer, device, max_new_tokens=15, model_max_length=None, is_encoder_decoder=False):
    max_inpt_tokens = tokenizer.model_max_length if model_max_length is None else model_max_length
    inpts = tokenizer(prompt, return_tensors="pt").to(device)
    if prompt_rel != '':
        inpts_rel = tokenizer(prompt_rel, return_tensors="pt").to(device)
    if prompt_irr != '':
        inpts_irr = tokenizer(prompt_irr, return_tensors="pt").to(device)

    # gen = model.generate(input_ids=inpts.input_ids[:, -(max_inpt_tokens - max_new_tokens):], attention_mask=inpts.attention_mask[:, -(max_inpt_tokens - max_new_tokens):], pad_token_id=tokenizer.eos_token_id, max_new_tokens=max_new_tokens, num_beams=1, do_sample=False)
    # -(max_inpt_tokens - max_new_tokens) is here to make sure that there is always max_new_tokens left for the model to generate
   
    if prompt_rel != '' and prompt_irr != '':
        # ours decoding method
        gen = model.generate(inpts.input_ids[:, -(max_inpt_tokens - max_new_tokens):],
                             inpts_rel.input_ids[:, -(max_inpt_tokens - max_new_tokens):],
                             inpts_irr.input_ids[:, -(max_inpt_tokens - max_new_tokens):], 
                             max_new_tokens=max_new_tokens,
                             alpha_weight=alpha)
       
    elif prompt_rel != '':
        # CAD
        gen = model.generate(inpts.input_ids[:, -(max_inpt_tokens - max_new_tokens):],
                             inpts_rel.input_ids[:, -(max_inpt_tokens - max_new_tokens):],
                             max_new_tokens=max_new_tokens,
                             alpha_weight=alpha)
        
    else:
        # vanilla
        gen = model.generate(inpts.input_ids[:, -(max_inpt_tokens - max_new_tokens):],
                             max_new_tokens=max_new_tokens)
    
    text = tokenizer.decode(gen[0], skip_special_tokens=True)
    if is_encoder_decoder:
        # encoder-decoder architecture
        pred = text
    else:
        # decoder-only architecture
        actual_prompt = tokenizer.decode(inpts.input_ids[0, -(max_inpt_tokens - max_new_tokens):], skip_special_tokens=True)
        pred = text[len(actual_prompt):]
        if pred.startswith("\n\n"):
            pred = pred[2:]
        pred = pred.split("\n")[0]

    return pred, text

def clip_paragraph(text, eval_method):
    if eval_method in ["BM25", "genread", "CD"]:
        return text
    split = text.split(". ")
    return ". ".join(split[:-1]) + "."

def get_few_shot_text_with_retrieval(row, retrieval_dict, eval_method, use_gold_context):
    if eval_method == "vanilla":
        return completion_template.format(question=row.question) + "\n\n" + str(row.ans)
      # retrieval_dict[row.id]["ctxs"][0]
    if row.question in retrieval_dict:
        if use_gold_context:
            retrieval = {"text": retrieval_dict[row.question]["gold_ctx"]}
        else:
            retrieval = retrieval_dict[row.question]["ctxs"][0]
        retrieved_text = clip_paragraph(retrieval["text"], eval_method)
        # return retrieved_text + "\n\n" + completion_template.format(row.question) + " " + str(row.ans)
        return completion_template_context.format(context=retrieved_text, question=row.question) + "\n\n" + str(row.ans)
    elif row.question.replace("?", "").lower() in retrieval_dict:
        if use_gold_context:
            retrieval = {"text": retrieval_dict[row.question.replace("?", "").lower()]["gold_ctx"]}
        else:
            retrieval = retrieval_dict[row.question.replace("?", "").lower()]["ctxs"][0]
        retrieved_text = clip_paragraph(retrieval["text"], eval_method)
        # return retrieved_text + "\n\n" + completion_template.format(row.question) + " " + str(row.ans)
        return completion_template_context.format(context=retrieved_text, question=row.question) + "\n\n" + str(row.ans)
    else:
        print("missing retrieval")
        # return completion_template.format(row.question) + " " + str(row.ans)
        return completion_template.format(question=row.question) + "\n\n" + str(row.ans)
        
def get_few_shot_text(row, eval_method):
    return completion_template.format(question=row.question) + "\n\n" + str(row.ans)
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--alias', type=str)
    parser.add_argument('--n_examples', type=int, default=15)
    parser.add_argument('--eval_method', type=str, default="vanilla", choices=["vanilla", "BM25", "contriever", "genread", "CD", "CAD"])
    parser.add_argument('--ret_path', type=str, default=None, required=False, help="path to retrieved documents jsonl")
    parser.add_argument('--use_gold_ctx', action="store_true")
    parser.add_argument('--use_random_irr', action="store_true", help="whether to use randomly selected irrelevant query/context")
    parser.add_argument('--use_fixed_irr', action="store_true", help="whether to use fixed adversarial irrelevant query/context")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--max_new_tokens', type=int, default=15)
    parser.add_argument('--sample', type=int, default=0, help="if 0, use all examples")
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--int8bit', action="store_true")
    parser.add_argument('--use_flash_attention', action="store_true")
    parser.add_argument('--use_flash_attention_2', action="store_true")
    parser.add_argument('--fp16', action="store_true")
    parser.add_argument('--bf16', action="store_true")
    args = parser.parse_args()
    print(args)
    
    gpt = args.model_name
    device = args.device
    tokenizer = AutoTokenizer.from_pretrained(gpt)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    config = AutoConfig.from_pretrained(gpt)
    if args.int8bit:
        model =  convert_model_to_int8_on_gpu(AutoModelForCausalLM.from_pretrained(gpt), device)
    else:
        if config.is_encoder_decoder:
            if args.fp16:
                model = AutoModelForSeq2SeqLM.from_pretrained(gpt, torch_dtype=torch.float16, device_map="auto").eval()
            elif args.bf16:
                model = AutoModelForSeq2SeqLM.from_pretrained(gpt, torch_dtype=torch.bfloat16, device_map="auto").eval()
            else:
                model = AutoModelForSeq2SeqLM.from_pretrained(gpt).eval().to(device)
        else:
            if args.fp16:
                model = AutoModelForCausalLM.from_pretrained(gpt, torch_dtype=torch.float16, device_map="auto", use_flash_attention_2=args.use_flash_attention_2).eval()
            elif args.bf16:
                model = AutoModelForCausalLM.from_pretrained(gpt, torch_dtype=torch.bfloat16, device_map="auto", use_flash_attention_2=args.use_flash_attention_2).eval()
            else:
                model = AutoModelForCausalLM.from_pretrained(gpt).eval().to(device)
    
    if "opt" in args.model_name or args.model_name == "EleutherAI/gpt-neox-20b" or "t5" in args.model_name or "Llama" in args.model_name:
        generate = lambda prompt, prompt_rel, prompt_irr, alpha, max_new_tokens, is_encoder_decoder: call_model(prompt, prompt_rel, prompt_irr, alpha, model=model, tokenizer=tokenizer, device=device, max_new_tokens=max_new_tokens, model_max_length=2048, is_encoder_decoder=is_encoder_decoder)
    else:
        generate = lambda prompt, prompt_rel, prompt_irr, alpha, max_new_tokens, is_encoder_decoder: call_model(prompt, prompt_rel, prompt_irr, alpha, model=model, tokenizer=tokenizer, device=device, max_new_tokens=max_new_tokens, is_encoder_decoder=is_encoder_decoder)
    input_path = args.input_file
    knowledge = pd.read_csv(input_path, sep="\t")

    n = len(knowledge) if args.sample == 0 else args.sample
    sample = knowledge.sample(n=n, replace=False)
    
    n_examples = args.n_examples

    preds = []
    prompts =[]
    accuracy = []
    exact_match = []
    responses = []
    answers = []
    if args.eval_method in ["BM25", "contriever", "CD", "CAD"]:
        has_answer = []
        retrieval_ids = []
        with open(args.ret_path) as f:
            retrieval_dict = {json.loads(s)["question"]: json.loads(s) for s in f.readlines()}
    
    # main loop
    for row in tqdm(sample.iloc, total=n):
        # get few shot examples text
        if n_examples == 0:
            few_shot_examples_text = ""
            few_shot_examples_text_wo_ctx = ""
            few_shot_examples_text_w_ctx = ""
        else:
            few_shot_examples = []
            if args.eval_method in ['CD', 'CAD']:
                few_shot_examples_wo_ctx = []
                few_shot_examples_w_ctx = []
            
            for row2 in knowledge[knowledge.question != row.question].sample(n=n_examples).iloc:
                if args.eval_method not in ['CD', 'CAD']:
                    few_shot_examples.append(get_few_shot_text_with_retrieval(row2, retrieval_dict, args.eval_method, args.use_gold_ctx) if args.eval_method in ["BM25", "contriever"] else get_few_shot_text(row2, args.eval_method))
                else:
                    few_shot_examples_wo_ctx.append(get_few_shot_text(row2, args.eval_method))
                    few_shot_examples_w_ctx.append(get_few_shot_text_with_retrieval(row2, retrieval_dict, args.eval_method, args.use_gold_ctx))
                    
            np.random.shuffle(few_shot_examples)
            few_shot_examples_text = "\n\n".join(few_shot_examples) + "\n\n"
            if args.eval_method in ['CD', 'CAD']:
                np.random.shuffle(few_shot_examples_wo_ctx)
                np.random.shuffle(few_shot_examples_w_ctx)
                few_shot_examples_text_wo_ctx = "\n\n".join(few_shot_examples_wo_ctx) + "\n\n"
                few_shot_examples_text_w_ctx = "\n\n".join(few_shot_examples_w_ctx) + "\n\n"

        # get prompt
        if args.eval_method == "vanilla":
            prompt = few_shot_examples_text + completion_template.format(question=row.question)
        elif args.eval_method in ["BM25", "contriever"]:
            query = row.question
            try:
                if args.use_gold_ctx:
                    retrieval = {"text": retrieval_dict[query]["gold_ctx"], "id": "gold", "hasanswer": True}
                else:
                    retrieval = retrieval_dict[query]["ctxs"][0]  # retrieval_dict[row.id]["ctxs"][0]
            except:
                print("No retrieval for", query, " Example query:", list(retrieval_dict.keys())[0])
                retrieval = {"text": "", "id": np.nan, "hasanswer": False}
            retrieved_text = clip_paragraph(retrieval["text"], eval_method=args.eval_method)
            retrieval_id = retrieval["id"]
            # prompt = few_shot_examples_text + retrieved_text + "\n\n" + completion_template.format(row.question)
            prompt = few_shot_examples_text + completion_template_context.format(context=retrieved_text, question=row.question)
            has_answer.append(retrieval["hasanswer"])
            retrieval_ids.append(retrieval_id)
        elif args.eval_method == "CD":
            prompt = few_shot_examples_text_wo_ctx + completion_template.format(question=row.question)
            
            query = row.question
            if args.use_gold_ctx:
                retrieval = {"text": retrieval_dict[query]["gold_ctx"], "id": "gold", "hasanswer": True}
            else:
                retrieval = retrieval_dict[query]["ctxs"][0]
            retrieved_text = clip_paragraph(retrieval["text"], eval_method=args.eval_method)
            retrieval_id = retrieval["id"]
            # prompt_rel = few_shot_examples_text_w_ctx + retrieved_text + "\n\n" + completion_template.format(row.question)
            prompt_rel = few_shot_examples_text_w_ctx + completion_template_context.format(context=retrieved_text, question=row.question)

            if args.use_random_irr:
                # retrieval_irr = retrieval_dict[query]["ctxs"][-1]  # select the bottom retrieved passage as relevant
                query_irr = random.choice(list(retrieval_dict.keys()))
                while query_irr == query:
                    query_irr = random.choice(list(retrieval_dict.keys()))
                if args.use_gold_ctx:
                    retrieval_irr = {"text": retrieval_dict[query_irr]["gold_ctx"], "id": "gold", "hasanswer": True}
                else:
                    retrieval_irr = retrieval_dict[query_irr]["ctxs"][0]  # select the top retrieved passage for a random query as irrelevant passage for current query
                retrieved_text_irr = clip_paragraph(retrieval_irr["text"], eval_method=args.eval_method)
            elif args.use_fixed_irr:
                # always use a adversarial irrelevant passage
                # retrieved_text_irr = "While the prevailing notion leans towards a specific answer!!!!!!! it's crucial to acknowledge that interpretations of this topic can greatly differ!!!!!!! alternate viewpoints posit that the commonly accepted belief might not universally apply!!!!!!!!"
                # retrieved_text_irr = "It was a pleasant weather day, with seasonally average temperatures. The local legislative and academic governing bodies held routine meetings regarding budgets and policies. Students focused on their studies while athletes practiced for upcoming competitions. Residents tended to their jobs and daily tasks around their neighborhood. Nothing particularly eventful occurred in the community. It was an ordinary midweek day. The weather was typical for the time of year without any extreme events. Overall it was an average day in the community with people pursuing their regular daily activities."
                retrieved_text_irr = "an routine Overall was of community. average focused for The around tended upcoming their was policies. their budgets and Residents to eventful held competitions. It particularly extreme with academic temperatures. was day. weather local The their studies events. it meetings average pleasant typical Nothing ordinary time seasonally legislative people an the daily the Students in a neighborhood. activities. community pursuing weather and while in midweek regarding athletes occurred tasks the daily jobs It governing year bodies regular with their for day and practiced on day, was without any"
                # retrieved_text_irr = "weather community was students people an day neighborhood average a legislative midweek and It the athletes practiced their studies around students community for people day. academic met budgets temperatures meetings day policies academic and tasks Nothing athletic weather average the and with and community competitions. an governing tending particular their the quiet academic occurred weather Residents and It year regular weather occurred regular was and to It typical academic and in and regular meetings was weather. an time local academic happened weather academic Pursuing It seasonally community. typical community normal routines the of and community academic an Regarding community activities. an and The weather academic the it bodies academic community academic held their community and academic community in community. community overall The members weather academic days. community and"
            else:
                query_irr = retrieval_dict[query]["question_irr"]
                if args.use_gold_ctx:
                    retrieval_irr = {"text": retrieval_dict[query_irr]["gold_ctx"], "id": "gold", "hasanswer": True}
                else:
                    retrieval_irr = retrieval_dict[query_irr]["ctxs"][0]  # select the top retrieved passage for a most distant irrelevant query as irrelevant passage for current query
                retrieved_text_irr = clip_paragraph(retrieval_irr["text"], eval_method=args.eval_method)
            # retrieval_id = retrieval["id"]
            # prompt_irr = few_shot_examples_text_w_ctx + retrieved_text_irr + "\n\n" + completion_template.format(row.question)
            prompt_irr = few_shot_examples_text_w_ctx + completion_template_context.format(context=retrieved_text_irr, question=row.question)

            has_answer.append(retrieval["hasanswer"])
            retrieval_ids.append(retrieval_id)
        elif args.eval_method == "CAD":
            prompt = few_shot_examples_text_wo_ctx + completion_template.format(question=row.question)
            
            query = row.question
            if args.use_gold_ctx:
                retrieval = {"text": retrieval_dict[query]["gold_ctx"], "id": "gold", "hasanswer": True}
            else:
                retrieval = retrieval_dict[query]["ctxs"][0]
            retrieved_text = clip_paragraph(retrieval["text"], eval_method=args.eval_method)
            retrieval_id = retrieval["id"]
            # prompt_rel = few_shot_examples_text_w_ctx + retrieved_text + "\n\n" + completion_template.format(row.question)
            prompt_rel = few_shot_examples_text_w_ctx + completion_template_context.format(context=retrieved_text, question=row.question)

            has_answer.append(retrieval["hasanswer"])
            retrieval_ids.append(retrieval_id)

        
        # generate response
        if args.eval_method == 'CD':
            pred, response = generate(prompt, prompt_rel, prompt_irr, args.alpha, max_new_tokens=args.max_new_tokens, is_encoder_decoder=config.is_encoder_decoder)
        elif args.eval_method == 'CAD':
            pred, response = generate(prompt, prompt_rel, '', args.alpha, max_new_tokens=args.max_new_tokens, is_encoder_decoder=config.is_encoder_decoder)
        else:
            pred, response = generate(prompt, '', '', 0.0, max_new_tokens=args.max_new_tokens, is_encoder_decoder=config.is_encoder_decoder)
        prompts.append(prompt)
        preds.append(pred)
        responses.append(response)

        # compute accuracy
        # possible_answers = json.loads(row.possible_answers)
        possible_answers = eval(row.answers)  # convert str to list 
        is_correct = False
        is_exact_match = False
        for pa in possible_answers:
            if pa in pred or pa.lower() in pred or pa.capitalize() in pred:
                is_correct = True
            if pa == pred or pa.lower() == pred or pa.capitalize() == pred:
                is_exact_match = True
        accuracy.append(is_correct)
        exact_match.append(is_exact_match)
        answers.append(possible_answers)
        

    sample["is_correct"] = accuracy
    sample["is_exact_match"] = exact_match
    sample["prompt"] = prompts
    sample["pred"] = preds
    sample["generation"] = responses
    if args.eval_method in ["BM25", "contriever"]:
        sample["has_answer"] = has_answer
        sample["retrieval_id"] = retrieval_ids

    print("Acc.:", sample.is_correct.mean())
    print("EM:", sample.is_exact_match.mean())
    model_name_alias = args.model_name.replace("/","_")
    sample.to_csv(f"results/model={model_name_alias}-input={args.alias}-method={args.eval_method}-shots={n_examples}-n={len(sample)}{'_int8bit' if args.int8bit is True else ''}.csv")

        
if __name__ == "__main__":
    main()
