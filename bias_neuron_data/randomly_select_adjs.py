import json
from random import sample, choice


neg_adj_vocab = json.load(open('neg_adj_vocab.json', 'r', encoding='utf-8'))
pos_adj_vocab = json.load(open('pos_adj_vocab.json', 'r', encoding='utf-8'))
current_neg_adjs = json.load(open('all_neg_adjs_v2.json', 'r', encoding='utf-8'))

selected_pos_adjs = sample(pos_adj_vocab, 100)
selected_neg_adjs = []
for _ in range(100-len(current_neg_adjs)):
    random_adj = choice(neg_adj_vocab)
    while random_adj in current_neg_adjs:
        random_adj = choice(neg_adj_vocab)
    selected_neg_adjs.append(random_adj)

with open('100_random_neg_adjs.json', 'w', encoding='utf-8') as nf, \
     open('100_random_pos_adjs.json', 'w', encoding='utf-8') as pf:
    json.dump(selected_neg_adjs, nf, indent=4)
    json.dump(selected_pos_adjs, pf, indent=4)

