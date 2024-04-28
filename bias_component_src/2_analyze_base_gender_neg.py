import json
from matplotlib import pyplot as plt
import numpy as np
import os
from collections import Counter
import random


bn_dir = '../base_component_results_ori_v2/bn/'
fig_dir = '../base_component_results_ori_v2/gender-neg/figs/'


# =========== stat bn_bag base ==============

tot_bag_num = 0
tot_rel_num = 0
tot_bneurons = 0
base_bn_bag_counter = Counter()
for filename in os.listdir(bn_dir):
    if not filename.startswith('base_bn_bag-'):
        continue
    with open(os.path.join(bn_dir, filename), 'r') as f:
        bn_bag_list = json.load(f)
        for bn_bag in bn_bag_list:
            for bn in bn_bag:
                base_bn_bag_counter.update([bn[0]])
        tot_bag_num += len(bn_bag_list)
for k, v in base_bn_bag_counter.items():
    tot_bneurons += base_bn_bag_counter[k]
for k, v in base_bn_bag_counter.items():
    base_bn_bag_counter[k] /= tot_bneurons
print('average base_bn', tot_bneurons / tot_bag_num)

# =========== plot base neuron distribution ===========

plt.figure(figsize=(8, 3))

x = np.array([i + 1 for i in range(12)])
y = np.array([base_bn_bag_counter[i] for i in range(12)])
plt.xlabel('Layer', fontsize=20)
plt.ylabel('Percentage', fontsize=20)
plt.xticks([i + 1 for i in range(12)], labels=[i + 1 for i in range(12)], fontsize=20)
plt.yticks(np.arange(-0.4, 0.5, 0.1), labels=[f'{np.abs(i)}%' for i in range(-40, 50, 10)], fontsize=13)
plt.tick_params(axis="y", left=False, right=True, labelleft=False, labelright=True, rotation=0, labelsize=18)
plt.ylim(-y.max() - 0.03, y.max() + 0.03)
plt.xlim(0.3, 12.7)
bottom = -y
y = y * 2
plt.bar(x, y, width=1.02, color='#0165fc', bottom=bottom)
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()

plt.savefig(os.path.join(fig_dir, 'base_distribution.pdf'), dpi=100)
plt.close()


# ========================================================================================
#                       base neuron intersection analysis
# ========================================================================================


def pos_list2str(pos_list):
    return '@'.join([str(pos) for pos in pos_list])


def pos_str2list(pos_str):
    return [int(pos) for pos in pos_str.split('@')]


def cal_intersec(bn_bag_1, bn_bag_2):
    bn_bag_1 = set(['@'.join(map(str, bn)) for bn in bn_bag_1])
    bn_bag_2 = set(['@'.join(map(str, bn)) for bn in bn_bag_2])
    return len(bn_bag_1.intersection(bn_bag_2))


# ====== load base bn =======
bn_bag_list_per_rel = {}
for filename in os.listdir(bn_dir):
    if not filename.startswith('base_bn_bag-'):
        continue
    with open(os.path.join(bn_dir, filename), 'r') as f:
        bn_bag_list = json.load(f)
    rel = filename.split('.')[0].split('-')[1]
    bn_bag_list_per_rel[rel] = bn_bag_list

# base inner
inner_ave_intersec = []
for rel, bn_bag_list in bn_bag_list_per_rel.items():
    print(f'calculating {rel}')
    len_bn_bag_list = len(bn_bag_list)
    for i in range(0, len_bn_bag_list):
        for j in range(i + 1, len_bn_bag_list):
            bn_bag_1 = bn_bag_list[i]
            bn_bag_2 = bn_bag_list[j]
            num_intersec = cal_intersec(bn_bag_1, bn_bag_2)
            inner_ave_intersec.append(num_intersec)
inner_ave_intersec = np.array(inner_ave_intersec).mean()
print(f'base bn has on average {inner_ave_intersec} inner bn interseciton')

# base inter
inter_ave_intersec = []
for rel, bn_bag_list in bn_bag_list_per_rel.items():
    print(f'calculating {rel}')
    len_bn_bag_list = len(bn_bag_list)
    for i in range(0, len_bn_bag_list):
        for j in range(0, 100):
            bn_bag_1 = bn_bag_list[i]
            other_rel = random.choice([x for x in bn_bag_list_per_rel.keys() if x != rel])
            other_idx = random.randint(0, len(bn_bag_list_per_rel[other_rel]) - 1)
            bn_bag_2 = bn_bag_list_per_rel[other_rel][other_idx]
            num_intersec = cal_intersec(bn_bag_1, bn_bag_2)
            inter_ave_intersec.append(num_intersec)
inter_ave_intersec = np.array(inter_ave_intersec).mean()
print(f'base bn has on average {inter_ave_intersec} inter bn interseciton')
