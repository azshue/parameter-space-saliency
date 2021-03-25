import os
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from transformers import BertForMaskedLM, BertTokenizer, AutoTokenizer, AutoModelForMaskedLM, AutoConfig
from datasets import load_dataset

from data import load_file, filter_samples, apply_template, batchify
from utils import sort_grads
from inference import get_dataset_stats, test_and_find_incorrect_prediction, sample_saliency_curves

CACHE_DIR = './cache'

def closest_k_saliency(reference_saliency_list, sample_saliency_list, k=10, lst=[], cosine=True):
    # Initialize list of indices to output
    idx_lst = []

    # Iterate through all the input indices
    for i in lst:
        if cosine:
            dist_vec = F.cosine_similarity(reference_saliency_list[i].repeat(sample_saliency_list.size()[0], 1), sample_saliency_list, dim=1)
            dist_vec = torch.abs(dist_vec - 1.0)
            dist_vec = dist_vec.cpu()
        else:
            dist_vec = F.pairwise_distance(reference_saliency_list[i].repeat(sample_saliency_list.size()[0], 1), sample_saliency_list)
        # print("distance vec size: ", dist_vec.size())
        # Use distance matrix to find the top k smallest distance elements away
        distances, closest_idx = torch.topk(dist_vec, k, largest=False)
        # print("closest index for sample{:d}: ".format(i), closest_idx)
        idx_lst.append([distances, closest_idx])
    
    return idx_lst


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Saliency curve comparison')
    parser.add_argument('--bert_model_name',
                        default='bert-base-cased', type=str)
    parser.add_argument('--dataset_name', default='/cmlscratch/manlis/data/LAMA/Squad/test.jsonl', type=str)
    parser.add_argument('--max_seq_length', default=1024, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--aggr', default="column-wise", type=str,
                        choices=['naive', 'column-wise'])
    # knn
    parser.add_argument('--k_closest', default=5, type=int, help='number of closest saliency curves')
    parser.add_argument('--chosen_samples', default="[1, 10, 100]", 
                            type=str, help='list of samples_ids to run k_closest on')
    args = parser.parse_args()

    if args.chosen_samples == "[]":
        args.chosen_samples = list(rng.integers(300, size=5))  #len(incorrect_id)
    else:
        string_list = args.chosen_samples
        lst = string_list.strip('][').split(',') 
        for i in range(len(lst)):
            lst[i] = int(lst[i])
        args.chosen_samples = lst
    print("slected sample ids: ", args.chosen_samples)

    # set up the model
    tokenizer = AutoTokenizer.from_pretrained(
        args.bert_model_name, cache_dir=CACHE_DIR)

    config = AutoConfig.from_pretrained(
        args.bert_model_name, cache_dir=CACHE_DIR)
    model = AutoModelForMaskedLM.from_pretrained(
        args.bert_model_name,
        config=config,
        cache_dir=CACHE_DIR,
    )
    model.resize_token_embeddings(len(tokenizer))

    # load data TODO: add an outer loop for relations
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

    # get inference results
    inference_file = './{:s}_inference_results_{:s}.pth'.format(
        Path(os.path.dirname(args.dataset_name)).stem, args.bert_model_name)
    if os.path.isfile(inference_file):
        inf_results = torch.load(inference_file)
        incorrect_id = inf_results['incorrect_id']
        incorrect_predictions = inf_results['incorrect_predictions']
        correct_id = inf_results['correct_id']
    else:
        incorrect_id, incorrect_predictions, correct_id = test_and_find_incorrect_prediction(
            all_samples, model, tokenizer, args)
        torch.save({'incorrect_id': incorrect_id,
                    'incorrect_predictions': incorrect_predictions,
                    'correct_id': correct_id}, inference_file)

    # mean and std of the testset
    stat_file = './{:s}_{:s}_saliency_stat_{:s}.pth'.format(
        Path(os.path.dirname(args.dataset_name)).stem, args.aggr, args.bert_model_name)
    if os.path.isfile(stat_file):
        stats = torch.load(stat_file)
        testset_mean = stats['testset_mean']
        testset_std = stats['testset_std']
    else:
        testset_mean, testset_std = get_dataset_stats(
            all_samples, model, tokenizer, args)
        torch.save({'testset_mean': testset_mean,
                    'testset_std': testset_std}, stat_file)

    # saliency curves
    saliency_file = './{:s}_{:s}_saliency_profile_{:s}.pth'.format(Path(os.path.dirname(args.dataset_name)).stem, args.aggr, args.bert_model_name)
    if os.path.isfile(saliency_file):
        saliency_curves = torch.load(saliency_file)
    else:
        saliency_curves = sample_saliency_curves(all_samples, model, tokenizer, testset_mean, testset_std, args)
        torch.save(saliency_curves, saliency_file)

    print(saliency_curves.size())

    closest_idx_lst = closest_k_saliency(saliency_curves, saliency_curves, args.k_closest, lst=args.chosen_samples)

    print(closest_idx_lst[:, -1])
    samples_batches, sentences_batches, label_batches = batchify(all_samples, 1)
    for ind, res in enumerate(closest_idx_lst):
        print("reference sample: {}".format(label_batches[args.chosen_samples[ind]]))
        nns = list(res[-1].numpy())
        for i, id in enumerate(nns):
            print("{} closest: {}".format(i, label_batches[id]))



