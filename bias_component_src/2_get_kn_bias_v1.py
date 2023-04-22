import jsonlines, json
import numpy as np
from collections import Counter
import os

import matplotlib.pyplot as plt
import seaborn as sns

threshold_ratio = 0.2
mode_ratio_bag = 0.7
mode_ratio_rel = 0.1
kn_dir = '../bias_component_results_ori_v1/kn/'
# rlts_dir = '../results/'
rlts_dir = '../bias_component_results_ori_v1/'


def re_filter(metric_triplets):
    metric_max = -999
    for i in range(len(metric_triplets)):
        metric_max = max(metric_max, metric_triplets[i][2])
    # 过滤使得 metric_triplets 中只包含 attr_score >= max(attr_score) * threshold 的 triplet
    metric_triplets = [triplet for triplet in metric_triplets if triplet[2] >= metric_max * threshold_ratio]
    return metric_triplets


def pos_list2str(pos_list):
    return '@'.join([str(pos) for pos in pos_list])


def pos_str2list(pos_str):
    return [int(pos) for pos in pos_str.split('@')]


def parse_kn(pos_cnt, tot_num, mode_ratio, min_threshold=0):
    mode_threshold = tot_num * mode_ratio
    # print('111 mode_threshold', mode_threshold)
    mode_threshold = max(mode_threshold, min_threshold)
    # print('222 mode_threshold', mode_threshold)
    kn_bag = []
    for pos_str, cnt in pos_cnt.items():
        if cnt >= mode_threshold:
            kn_bag.append(pos_str2list(pos_str))
    return kn_bag


# def analysis_file(filename, metric='ig_gold'):
def analysis_file(filename, metric='ig_gold_gap'):
    # print('filename', filename)
    rel = filename.split('.')[0].split('-')[-1]
    print(f'===========> parsing important position in {rel}..., mode_ratio_bag={mode_ratio_bag}')

    rlts_bag_list = []
    with open(os.path.join(rlts_dir, filename), 'r') as fr:
        for rlts_bag in jsonlines.Reader(fr):
            # print('bag len', len(rlts_bag))  # 4: 是个长度为4的dict, keys: 'pred','ig_pred','ig_gold','base'
            # print('rlts_bag 0', rlts_bag[0])
            rlts_bag_list.append(rlts_bag)

    ave_kn_num = 0

    kn_bag_list = []
    # get imp pos by bag_ig
    for bag_idx, rlts_bag in enumerate(rlts_bag_list):
        pos_cnt_bag = Counter()
        for rlt in rlts_bag:
            # print('rlt', rlt)
            res_dict = rlt[1]
            # print('res_dict keys', res_dict.keys())
            metric_triplets = re_filter(res_dict[metric])
            # print('metric_triplets', metric_triplets)
            for metric_triplet in metric_triplets:
                pos_cnt_bag.update([pos_list2str(metric_triplet[:2])])
        # print('pos_cnt_bag', pos_cnt_bag)
        # pos_cnt_bag统计neuron出现的次数, parse_kn过滤得到出现次数超过threshold(3)的neuron
        kn_bag = parse_kn(pos_cnt_bag, len(rlts_bag), mode_ratio_bag, 3)  # 过滤当前bag, 也就是一条fact, 用的是bag ratio
        # print('kn_bag len', len(kn_bag)) 
        ave_kn_num += len(kn_bag)
        kn_bag_list.append(kn_bag)

    # print('rlts_bag_list len', len(rlts_bag_list))  # 977
    ave_kn_num /= len(rlts_bag_list)

    # get imp pos by rel_ig
    pos_cnt_rel = Counter()
    for kn_bag in kn_bag_list:
        for kn in kn_bag:
            pos_cnt_rel.update([pos_list2str(kn)])
    kn_rel = parse_kn(pos_cnt_rel, len(kn_bag_list), mode_ratio_rel)  # 所有bag, 也就是当前relation, 所以用的是rel ratio

    return ave_kn_num, kn_bag_list, kn_rel


def stat(data, pos_type, rel):
    if pos_type == 'kn_rel':
        print(f'{rel}\'s {pos_type} has {len(data)} imp pos. ')
        return
    ave_len = 0
    for kn_bag in data:
        ave_len += len(kn_bag)
    ave_len /= len(data)
    print(f'{rel}\'s {pos_type} has on average {ave_len} imp pos. ')


if not os.path.exists(kn_dir):
    os.makedirs(kn_dir)
for filename in os.listdir(rlts_dir):
    # if filename.endswith('.rlt.jsonl'):
    if filename.endswith('.rlt.jsonl') and 'filtered-gap-rm-base' in filename:
        # print('correct filename', filename)
        threshold_ratio = 0.2
        mode_ratio_bag = 0.7
        for max_it in range(6):
            ave_kn_num, kn_bag_list, kn_rel = analysis_file(filename)  # ig_gold_gap
            if ave_kn_num < 2:
                mode_ratio_bag -= 0.05
            if ave_kn_num > 5:
                mode_ratio_bag += 0.05
            if ave_kn_num >= 2 and ave_kn_num <= 5:
                break
        rel = filename.split('.')[0].split('-')[-1]
        stat(kn_bag_list, 'kn_bag', rel)
        stat(kn_rel, 'kn_rel', rel)
        with open(os.path.join(kn_dir, f'kn_bag-{rel}.json'), 'w') as fw:
            json.dump(kn_bag_list, fw, indent=2)
        with open(os.path.join(kn_dir, f'kn_rel-{rel}.json'), 'w') as fw:
            json.dump(kn_rel, fw, indent=2)

        # threshold_ratio = 0.5
        # mode_ratio_bag = 0.7
        # for max_it in range(6):
        #     ave_kn_num, kn_bag_list, kn_rel = analysis_file(filename, 'base')
        #     if ave_kn_num < 2:
        #         mode_ratio_bag -= 0.05
        #     if ave_kn_num > 5:
        #         mode_ratio_bag += 0.05
        #     if ave_kn_num >= 2 and ave_kn_num <= 5:
        #         break
        # rel = filename.split('.')[0].split('-')[-1]
        # stat(kn_bag_list, 'kn_bag', rel)
        # stat(kn_rel, 'kn_rel', rel)
        # with open(os.path.join(kn_dir, f'base_kn_bag-{rel}.json'), 'w') as fw:
        #     json.dump(kn_bag_list, fw, indent=2)
        # with open(os.path.join(kn_dir, f'base_kn_rel-{rel}.json'), 'w') as fw:
        #     json.dump(kn_rel, fw, indent=2)
