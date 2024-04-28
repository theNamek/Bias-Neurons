import logging
import argparse
import math
import os
import torch
import random
import numpy as np
import json, jsonlines
import time
import jsonlines

from transformers import BertTokenizer
from custom_bert_bias import BertForMaskedLM
import torch.nn.functional as F

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def example2feature(example, max_seq_length, tokenizer):
    ori_tokens = tokenizer.tokenize(example[0])
    if len(ori_tokens) > max_seq_length - 2:
        ori_tokens = ori_tokens[:max_seq_length - 2]

    tokens = ["[CLS]"] + ori_tokens + ["[SEP]"]
    base_tokens = ["[UNK]"] + ["[UNK]"] * len(ori_tokens) + ["[UNK]"]
    segment_ids = [0] * len(tokens)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    baseline_ids = tokenizer.convert_tokens_to_ids(base_tokens)  # unk id
    input_mask = [1] * len(input_ids)

    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    baseline_ids += padding
    segment_ids += padding
    input_mask += padding

    assert len(baseline_ids) == max_seq_length
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    features = {
        'input_ids': input_ids,
        'input_mask': input_mask,
        'segment_ids': segment_ids,
        'baseline_ids': baseline_ids,
    }
    tokens_info = {
        "tokens":tokens,
        "relation":example[2],
        "gold_obj":example[1],
        "pred_obj": None
    }
    return features, tokens_info


def scaled_input(emb, batch_size, num_batch):
    baseline = torch.zeros_like(emb)

    num_points = batch_size * num_batch
    step = (emb - baseline) / num_points

    res = torch.cat([torch.add(baseline, step * i) for i in range(num_points)], dim=0)
    return res, step[0]


def convert_to_triplet_ig2(ig2_list):
    ig2_triplet = []
    ig2 = np.array(ig2_list)
    max_ig2 = ig2.max()
    for i in range(ig2.shape[0]):
        for j in range(ig2.shape[1]):
            if ig2[i][j] >= max_ig2 * 0.1:
                ig2_triplet.append([i, j, ig2[i][j]])
    return ig2_triplet


def main():
    parser = argparse.ArgumentParser()

    # Basic parameters
    parser.add_argument("--data_path",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data path. Should be .json file for the MLM task. ")
    parser.add_argument("--demographic_dimension",
                        default=None,
                        type=str,
                        required=True,
                        help="The demographic dimension, such as Ethnicity, Gender, etc.")
    parser.add_argument("--demographic1",
                        default=None,
                        type=str,
                        required=True,
                        help="The first demographic for the gap computation, such as black.")
    parser.add_argument("--demographic2",
                        default=None,
                        type=str,
                        required=True,
                        help="The second demographic for the gap computation, such as white.")
    parser.add_argument("--modifier",
                        default=None,
                        type=str,
                        required=True,
                        help="The modifier type for the gap computation, such as negative (N).")
    # parser.add_argument("--tmp_data_path",
    #                     default=None,
    #                     type=str,
    #                     help="Temporary input data path. Should be .json file for the MLM task. ")
    # parser.add_argument("--demo1_data_path",
    #                     default=None,
    #                     type=str,
    #                     required=True,
    #                     help="The input data path for demo1 ethnicity. Should be .json file for the MLM task. ")
    # parser.add_argument("--demo1_tmp_data_path",
    #                     default=None,
    #                     type=str,
    #                     help="Temporary input data path for demo1 ethnicity. Should be .json file for the MLM task. ")
    # parser.add_argument("--demo2_data_path",
    #                     default=None,
    #                     type=str,
    #                     required=True,
    #                     help="The input data path for demo2 ethnicity. Should be .json file for the MLM task. ")
    # parser.add_argument("--demo2_tmp_data_path",
    #                     default=None,
    #                     type=str,
    #                     help="Temporary input data path for demo2 ethnicity. Should be .json file for the MLM task. ")
    # parser.add_argument("--bert_model", default=None, type=str, required=True,
    #                     help="Bert pre-trained model selected in the list: bert-base-uncased, "
    #                         "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--bert_model_path", default=None, type=str, required=True,
                        help="Self-defined path of Bert pre-trained model.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    # parser.add_argument("--demo1_output_dir",
    #                     default=None,
    #                     type=str,
    #                     required=True,
    #                     help="The output directory where the model predictions and checkpoints will be written.")
    # parser.add_argument("--demo2_output_dir",
    #                     default=None,
    #                     type=str,
    #                     required=True,
    #                     help="The output directory where the model predictions and checkpoints will be written.")
    # parser.add_argument("--output_prefix",
    #                     default=None,
    #                     type=str,
    #                     required=True,
    #                     help="The output prefix to indentify each running of experiment. ")

    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                            "Sequences longer than this will be truncated, and sequences shorter \n"
                            "than this will be padded.")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Set this flag if you are using an uncased model")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--gpus",
                        type=str,
                        default='0',
                        help="available gpus id")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument("--debug",
                        type=int,
                        default=-1,
                        help="How many examples to debug. -1 denotes no debugging")
    # parser.add_argument("--pt_relation",
    #                     type=str,
    #                     default=None,
    #                     help="Relation to calculate on clusters")

    # parameters about integrated grad
    parser.add_argument("--get_pred",
                        action='store_true',
                        help="Whether to get prediction results.")
    parser.add_argument("--get_ig2_pred",
                        action='store_true',
                        help="Whether to get integrated gradient at the predicted label.")
    parser.add_argument("--get_ig2_gold",
                        action='store_true',
                        help="Whether to get integrated gradient at the gold label.")
    parser.add_argument("--get_ig2_gold_filtered",
                        action='store_true',
                        help="Whether to get integrated gradient at the gold label after filtering.")
    parser.add_argument("--get_base",
                        action='store_true',
                        help="Whether to get base values. ")
    # parser.add_argument("--get_base_filtered",
    #                     action='store_true',
    #                     help="Whether to get base values after filtering. ")
    parser.add_argument("--get_ig2_gold_gap_filtered",
                        action='store_true',
                        help="Whether to get integrated gradient gap at the gold label after filtering.")
    # parser.add_argument("--get_base_gap_filtered",
    #                     action='store_true',
    #                     help="Whether to get base values gap after filtering. ")
    parser.add_argument("--batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for cut.")
    parser.add_argument("--num_batch",
                        default=10,
                        type=int,
                        help="Num batch of an example.")

    args = parser.parse_args()

    if args.no_cuda or not torch.cuda.is_available():
        device = torch.device("cpu")
        n_gpu = 0
    elif len(args.gpus) == 1:
        device = torch.device("cuda:%s" % args.gpus)
        n_gpu = 1
    print("device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, bool(n_gpu > 1)))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    demo1_data_path = os.path.join(args.data_path, args.demographic_dimension,
                                   args.demographic1 + '_' + args.modifier + '_data.json')
    demo2_data_path = os.path.join(args.data_path, args.demographic_dimension,
                                   args.demographic2 + '_' + args.modifier + '_data.json')

    demo1_tmp_data_path = os.path.join(args.data_path, args.demographic_dimension,
                                       args.demographic1 + '_' + args.modifier + '_allbags.json')
    demo2_tmp_data_path = os.path.join(args.data_path, args.demographic_dimension,
                                       args.demographic2 + '_' + args.modifier + '_allbags.json')

    output_prefix = 'Modifier-' + args.demographic_dimension + '-' + args.modifier

    json.dump(args.__dict__, open(os.path.join(args.output_dir, output_prefix + '.args.json'), 'w'), sort_keys=True, indent=2)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model_path, do_lower_case=args.do_lower_case)

    print("***** CUDA.empty_cache() *****")
    torch.cuda.empty_cache()
    model = BertForMaskedLM.from_pretrained(args.bert_model_path)
    model.to(device)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.eval()

    # demo1
    if os.path.exists(demo1_tmp_data_path):
        with open(demo1_tmp_data_path, 'r') as f:
            demo1_eval_bag_list_perrel = json.load(f)
    else:
        with open(demo1_data_path, 'r') as f:
            demo1_eval_bag_list_all = json.load(f)
        demo1_eval_bag_list_perrel = {}
        for bag_idx, eval_bag in enumerate(demo1_eval_bag_list_all):
            bag_rel = eval_bag[0][2].split('(')[0]
            if bag_rel not in demo1_eval_bag_list_perrel:
                demo1_eval_bag_list_perrel[bag_rel] = []
            if len(demo1_eval_bag_list_perrel[bag_rel]) >= args.debug:
                continue
            demo1_eval_bag_list_perrel[bag_rel].append(eval_bag)
        with open(demo1_tmp_data_path, 'w') as fw:
            json.dump(demo1_eval_bag_list_perrel, fw, indent=2)

    # demo2
    if os.path.exists(demo2_tmp_data_path):
        with open(demo2_tmp_data_path, 'r') as f:
            demo2_eval_bag_list_perrel = json.load(f)
    else:
        with open(demo2_data_path, 'r') as f:
            demo2_eval_bag_list_all = json.load(f)
        demo2_eval_bag_list_perrel = {}
        for bag_idx, eval_bag in enumerate(demo2_eval_bag_list_all):
            bag_rel = eval_bag[0][2].split('(')[0]
            if bag_rel not in demo2_eval_bag_list_perrel:
                demo2_eval_bag_list_perrel[bag_rel] = []
            if len(demo2_eval_bag_list_perrel[bag_rel]) >= args.debug:
                continue
            demo2_eval_bag_list_perrel[bag_rel].append(eval_bag)
        with open(demo2_tmp_data_path, 'w') as fw:
            json.dump(demo2_eval_bag_list_perrel, fw, indent=2)

    for relation, eval_bag_list in demo1_eval_bag_list_perrel.items():
        tic = time.perf_counter()
        with jsonlines.open(os.path.join(args.output_dir, output_prefix + '-' + args.demographic1 + '.rlt' + '.jsonl'), 'w') as demo1_fw:
            for bag_idx, eval_bag in enumerate(eval_bag_list):
                res_dict_bag = []
                for eval_example in eval_bag:
                    eval_features, tokens_info = example2feature(eval_example, args.max_seq_length, tokenizer)
                    baseline_ids, input_ids, input_mask, segment_ids = eval_features['baseline_ids'], eval_features['input_ids'], eval_features['input_mask'], eval_features['segment_ids']
                    baseline_ids = torch.tensor(baseline_ids, dtype=torch.long).unsqueeze(0)
                    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
                    input_mask = torch.tensor(input_mask, dtype=torch.long).unsqueeze(0)
                    segment_ids = torch.tensor(segment_ids, dtype=torch.long).unsqueeze(0)
                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    segment_ids = segment_ids.to(device)

                    tgt_pos = tokens_info['tokens'].index('[MASK]')

                    res_dict = {
                        'pred': [],
                        'ig2_pred': [],
                        'ig2_gold': [],
                        'base': []
                    }

                    if args.get_pred:
                        _, logits = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_pos, tgt_layer=0)  # (1, n_vocab)
                        base_pred_prob = F.softmax(logits, dim=1)  # (1, n_vocab)
                        res_dict['pred'].append(base_pred_prob.tolist())

                    for tgt_layer in range(model.bert.config.num_hidden_layers):
                        ffn_weights, logits = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_pos, tgt_layer=tgt_layer)  # (1, ffn_size), (1, n_vocab)
                        pred_label = int(torch.argmax(logits[0, :]))  # scalar, 这里获得了pred_label，方便后面计算ig2_pred
                        gold_label = tokenizer.convert_tokens_to_ids(tokens_info['gold_obj'])
                        tokens_info['pred_obj'] = tokenizer.convert_ids_to_tokens(pred_label)
                        scaled_weights, weights_step = scaled_input(ffn_weights, args.batch_size, args.num_batch)  # (num_points, ffn_size), (ffn_size)
                        scaled_weights.requires_grad_(True)

                        if args.get_ig2_pred:
                            ig2_pred = None
                            for batch_idx in range(args.num_batch):
                                batch_weights = scaled_weights[batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size]
                                _, grad = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_pos, tgt_layer=tgt_layer, tmp_score=batch_weights, tgt_label=pred_label)  # (batch, n_vocab), (batch, ffn_size)
                                grad = grad.sum(dim=0)  # (ffn_size)
                                ig2_pred = grad if ig2_pred is None else torch.add(ig2_pred, grad)  # (ffn_size)
                            ig2_pred = ig2_pred * weights_step  # (ffn_size)
                            res_dict['ig2_pred'].append(ig2_pred.tolist())

                        if args.get_ig2_gold:
                            ig2_gold = None
                            for batch_idx in range(args.num_batch):
                                batch_weights = scaled_weights[batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size]
                                _, grad = model(input_ids=input_ids, attention_mask=input_mask,
                                                token_type_ids=segment_ids, tgt_pos=tgt_pos, tgt_layer=tgt_layer,
                                                tmp_score=batch_weights, tgt_label=gold_label)
                                grad = grad.sum(dim=0)
                                ig2_gold = grad if ig2_gold is None else torch.add(ig2_gold, grad)
                            ig2_gold = ig2_gold * weights_step
                            res_dict['ig2_gold'].append(ig2_gold.tolist())

                        if args.get_base:
                            res_dict['base'].append(ffn_weights.squeeze().tolist())

                    if args.get_ig2_gold_filtered:
                        res_dict['ig2_gold'] = convert_to_triplet_ig2(res_dict['ig2_gold'])

                    res_dict_bag.append([tokens_info, res_dict])

                demo1_fw.write(res_dict_bag)

        toc = time.perf_counter()
        print(f"***** Relation: {relation} evaluated. Costing time: {toc - tic:0.4f} seconds *****")

    for relation, eval_bag_list in demo2_eval_bag_list_perrel.items():
        tic = time.perf_counter()
        with jsonlines.open(os.path.join(args.output_dir, output_prefix + '-' + args.demographic2 + '.rlt' + '.jsonl'), 'w') as demo2_fw:
            for bag_idx, eval_bag in enumerate(eval_bag_list):
                res_dict_bag = []
                for eval_example in eval_bag:
                    eval_features, tokens_info = example2feature(eval_example, args.max_seq_length, tokenizer)
                    baseline_ids, input_ids, input_mask, segment_ids = eval_features['baseline_ids'], eval_features['input_ids'], eval_features['input_mask'], eval_features['segment_ids']
                    baseline_ids = torch.tensor(baseline_ids, dtype=torch.long).unsqueeze(0)
                    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
                    input_mask = torch.tensor(input_mask, dtype=torch.long).unsqueeze(0)
                    segment_ids = torch.tensor(segment_ids, dtype=torch.long).unsqueeze(0)
                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    segment_ids = segment_ids.to(device)

                    tgt_pos = tokens_info['tokens'].index('[MASK]')

                    res_dict = {
                        'pred': [],
                        'ig2_pred': [],
                        'ig2_gold': [],
                        'base': []
                    }

                    if args.get_pred:
                        _, logits = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_pos, tgt_layer=0)
                        base_pred_prob = F.softmax(logits, dim=1)
                        res_dict['pred'].append(base_pred_prob.tolist())

                    for tgt_layer in range(model.bert.config.num_hidden_layers):
                        ffn_weights, logits = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_pos, tgt_layer=tgt_layer)
                        pred_label = int(torch.argmax(logits[0, :]))
                        gold_label = tokenizer.convert_tokens_to_ids(tokens_info['gold_obj'])
                        tokens_info['pred_obj'] = tokenizer.convert_ids_to_tokens(pred_label)
                        scaled_weights, weights_step = scaled_input(ffn_weights, args.batch_size, args.num_batch)
                        scaled_weights.requires_grad_(True)

                        if args.get_ig2_pred:
                            ig2_pred = None
                            for batch_idx in range(args.num_batch):
                                batch_weights = scaled_weights[batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size]
                                _, grad = model(input_ids=input_ids, attention_mask=input_mask,
                                                token_type_ids=segment_ids, tgt_pos=tgt_pos, tgt_layer=tgt_layer,
                                                tmp_score=batch_weights, tgt_label=pred_label)
                                grad = grad.sum(dim=0)
                                ig2_pred = grad if ig2_pred is None else torch.add(ig2_pred, grad)
                            ig2_pred = ig2_pred * weights_step
                            res_dict['ig2_pred'].append(ig2_pred.tolist())

                        if args.get_ig2_gold:
                            ig2_gold = None
                            for batch_idx in range(args.num_batch):
                                batch_weights = scaled_weights[batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size]
                                _, grad = model(input_ids=input_ids, attention_mask=input_mask,
                                                token_type_ids=segment_ids, tgt_pos=tgt_pos, tgt_layer=tgt_layer,
                                                tmp_score=batch_weights, tgt_label=gold_label)
                                grad = grad.sum(dim=0)
                                ig2_gold = grad if ig2_gold is None else torch.add(ig2_gold, grad)
                            ig2_gold = ig2_gold * weights_step
                            res_dict['ig2_gold'].append(ig2_gold.tolist())

                        if args.get_base:
                            res_dict['base'].append(ffn_weights.squeeze().tolist())

                    if args.get_ig2_gold_filtered:
                        res_dict['ig2_gold'] = convert_to_triplet_ig2(res_dict['ig2_gold'])

                    res_dict_bag.append([tokens_info, res_dict])

                demo2_fw.write(res_dict_bag)

        toc = time.perf_counter()
        print(f"***** Relation: {relation} evaluated. Costing time: {toc - tic:0.4f} seconds *****")

    for demo1_relation, demo2_relation in zip(demo1_eval_bag_list_perrel.keys(), demo2_eval_bag_list_perrel.keys()):
        with jsonlines.open(os.path.join(args.output_dir, output_prefix + '-' + args.demographic1 + '.rlt' + '.jsonl'), 'r') as fb, \
             jsonlines.open(os.path.join(args.output_dir, output_prefix + '-' + args.demographic2 + '.rlt' + '.jsonl'), 'r') as fw, \
             jsonlines.open(os.path.join(args.output_dir, output_prefix + '-filtered-gap-rm-base-' + args.demographic1 + '-' + args.demographic2 + '.rlt' + '.jsonl'), 'w') as filf_rmb_gap:
            tic = time.perf_counter()
            for demo1_res_dict_bag, demo2_res_dict_bag in zip(fb, fw):
                gap_res_dict_rmb_bag = []
                for demo1_example, demo2_example in zip(demo1_res_dict_bag, demo2_res_dict_bag):
                    demo1_res_dict, demo2_res_dict = demo1_example[1], demo2_example[1]
                    demo1_tokens_info, demo2_tokens_info = demo1_example[0], demo2_example[0]
                    demo1_ig2_gold = np.array(demo1_res_dict['ig2_gold'], np.float32)
                    demo1_base = np.array(demo1_res_dict['base'], np.float32)
                    demo2_ig2_gold = np.array(demo2_res_dict['ig2_gold'], np.float32)
                    demo2_base = np.array(demo2_res_dict['base'], np.float32)
                    ig2_gold_gap = demo1_ig2_gold - demo2_ig2_gold
                    ig2_gold_gap = ig2_gold_gap.tolist()
                    gap_tokens_info = {
                        "tokens": demo1_tokens_info['tokens'],
                        "gap_relation": demo1_relation + '-' + demo2_relation,
                        "gold_obj": "Black - White",
                    }
                    gap_res_rmb_dict = {
                        'ig2_gold_gap': ig2_gold_gap
                    }
                    if args.get_ig2_gold_gap_filtered:
                        gap_res_rmb_dict['ig2_gold_gap'] = convert_to_triplet_ig2(gap_res_rmb_dict['ig2_gold_gap'])
                    gap_res_dict_rmb_bag.append([gap_tokens_info, gap_res_rmb_dict])

                filf_rmb_gap.write(gap_res_dict_rmb_bag)

            toc = time.perf_counter()
            print(f"***** Relation: {demo1_relation + '-' + demo2_relation} evaluated. Costing time: {toc - tic:0.4f} seconds *****")


if __name__ == "__main__":
    main()
