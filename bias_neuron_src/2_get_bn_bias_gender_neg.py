import jsonlines, json
from collections import Counter
import os


threshold_ratio = 0.2
mode_ratio_bag = 0.7
mode_ratio_rel = 0.1
bn_dir = '../bias_component_results_ori_v2/bn/'
rlts_dir = '../bias_component_results_ori_v2/gender-neg'


def re_filter(metric_triplets):
    metric_max = -999
    for i in range(len(metric_triplets)):
        metric_max = max(metric_max, metric_triplets[i][2])
    metric_triplets = [triplet for triplet in metric_triplets if triplet[2] >= metric_max * threshold_ratio]
    return metric_triplets


def pos_list2str(pos_list):
    return '@'.join([str(pos) for pos in pos_list])


def pos_str2list(pos_str):
    return [int(pos) for pos in pos_str.split('@')]


def parse_bn(pos_cnt, tot_num, mode_ratio, min_threshold=0):
    mode_threshold = tot_num * mode_ratio
    mode_threshold = max(mode_threshold, min_threshold)
    bn_bag = []
    for pos_str, cnt in pos_cnt.items():
        if cnt >= mode_threshold:
            bn_bag.append(pos_str2list(pos_str))
    return bn_bag


def analysis_file(filename, metric='ig_gold_gap'):
    rel = filename.split('.')[0].split('-base-')[-1]
    print(f'===========> parsing important position in {rel}..., mode_ratio_bag={mode_ratio_bag}')

    rlts_bag_list = []
    with open(os.path.join(rlts_dir, filename), 'r') as fr:
        for rlts_bag in jsonlines.Reader(fr):
            rlts_bag_list.append(rlts_bag)

    ave_bn_num = 0

    bn_bag_list = []
    # get imp pos by bag_ig
    for bag_idx, rlts_bag in enumerate(rlts_bag_list):
        pos_cnt_bag = Counter()
        for rlt in rlts_bag:
            res_dict = rlt[1]
            metric_triplets = re_filter(res_dict[metric])
            for metric_triplet in metric_triplets:
                pos_cnt_bag.update([pos_list2str(metric_triplet[:2])])
        bn_bag = parse_bn(pos_cnt_bag, len(rlts_bag), mode_ratio_bag, 3)
        ave_bn_num += len(bn_bag)
        bn_bag_list.append(bn_bag)

    ave_bn_num /= len(rlts_bag_list)

    # get imp pos by rel_ig
    pos_cnt_rel = Counter()
    for bn_bag in bn_bag_list:
        for bn in bn_bag:
            pos_cnt_rel.update([pos_list2str(bn)])
    bn_rel = parse_bn(pos_cnt_rel, len(bn_bag_list), mode_ratio_rel)

    return ave_bn_num, bn_bag_list, bn_rel


def stat(data, pos_type, rel):
    if pos_type == 'bn_rel':
        print(f'{rel}\'s {pos_type} has {len(data)} imp pos. ')
        return
    ave_len = 0
    for bn_bag in data:
        ave_len += len(bn_bag)
    ave_len /= len(data)
    print(f'{rel}\'s {pos_type} has on average {ave_len} imp pos. ')


if not os.path.exists(bn_dir):
    os.makedirs(bn_dir)
for filename in os.listdir(rlts_dir):
    if filename.endswith('.rlt.jsonl') and 'filtered-gap-rm-base' in filename:
        threshold_ratio = 0.2
        mode_ratio_bag = 0.7
        for max_it in range(6):
            ave_bn_num, bn_bag_list, bn_rel = analysis_file(filename)  # ig_gold_gap
            if ave_bn_num < 2:
                mode_ratio_bag -= 0.05
            if ave_bn_num > 5:
                mode_ratio_bag += 0.05
            if ave_bn_num >= 2 and ave_bn_num <= 5:
                break
        rel = filename.split('.')[0].split('-base-')[-1]
        stat(bn_bag_list, 'bn_bag', rel)
        stat(bn_rel, 'bn_rel', rel)
        with open(os.path.join(bn_dir, f'bn_bag-{rel}.json'), 'w') as fw:
            json.dump(bn_bag_list, fw, indent=2)
        with open(os.path.join(bn_dir, f'bn_rel-{rel}.json'), 'w') as fw:
            json.dump(bn_rel, fw, indent=2)

