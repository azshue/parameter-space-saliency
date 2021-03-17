import os
import argparse

import torch
import transformers
from transformers import BertForMaskedLM, BertTokenizer, AutoTokenizer, AutoModelForMaskedLM, AutoConfig
from datasets import load_dataset

from data import load_file, filter_samples, apply_template, batchify
from utils import sort_grads

CACHE_DIR = './cache'

def inference(args):
    ### set up the model
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model_name, cache_dir=CACHE_DIR)

    config = AutoConfig.from_pretrained(args.bert_model_name, cache_dir=CACHE_DIR)
    model =  AutoModelForMaskedLM.from_pretrained(
            args.bert_model_name,
            config=config,
            cache_dir=CACHE_DIR,
        )
    model.resize_token_embeddings(len(tokenizer))

    ### load data TODO: add an outer loop for relations
    data = load_file(args.dataset_name)
    print(len(data))

    template = ""  # TODO: add a lookup dict for relation-template pairs
    all_samples, ret_msg = filter_samples(
        model, tokenizer, data, args.max_seq_length, template
    )
    print(ret_msg)
    print(len(all_samples))

    if template != "":
        all_samples = apply_template(all_samples, template)

    # create uuid if not present
    i = 0
    for sample in all_samples:
        if "uuid" not in sample:
            sample["uuid"] = i
        i += 1

    ### run inference
    samples_batches, sentences_batches, label_batches = batchify(all_samples, args.batch_size)
    for i in range(len(samples_batches)):
        samples_b = samples_batches[i]
        sentences_b = sentences_batches[i]
        gt_b = label_batches[i]

        inputs_b = tokenizer(sentences_b, return_tensors='pt')
        labels_b = tokenizer(gt_b, return_tensors='pt')["input_ids"]

        outputs = model(**inputs_b, labels=labels_b)
        loss = outputs.loss
        logits = outputs.logits

        # compute gradient profile
        # gradients = torch.autograd.grad(loss, model.parameters(), create_graph=False, allow_unused = True)
        loss.backward()

        saliency_profile = sort_grads(model, args.aggr)
            
        

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Saliency curve comparison')
    parser.add_argument('--bert_model_name', default='bert-base-cased', type=str)
    parser.add_argument('--dataset_name', default='/cmlscratch/manlis/data/LAMA/Squad/test.jsonl', type=str)
    parser.add_argument('--max_seq_length', default=1024, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    args = parser.parse_args()

    inference(args)