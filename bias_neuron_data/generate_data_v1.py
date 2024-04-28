import json

## ethnicity-neg
# black_template = json.load(open('bias_neuron_data_ethnicity_neg_black_template1.json', 'r'))
# white_template = json.load(open('bias_neuron_data_ethnicity_neg_white_template1.json', 'r'))
all_neg_adjs = json.load(open('all_neg_adjs.json', 'r'))
different_expressions = json.load(open('expression_gather_ethnicity_neg.json', 'r'))

black_data_save_path = 'data_ethnicity_neg_black.json'
white_data_save_path = 'data_ethnicity_neg_white.json'
all_black_data, all_white_data = [], []
id_count = 0
for neg_adj in all_neg_adjs:
    cur_adj_black_data, cur_adj_white_data = [], []
    id_count += 1
    for expression in different_expressions:
        sent_template = expression[0]
        generated_sent = sent_template.replace('[Modifier]', neg_adj)
        # print('neg-'+neg_adj)
        sample_id = expression[2].split('(')[0]
        print('sample_id', sample_id)
        if id_count < 10:
            black_sample_id = 'BN0' + str(id_count)
            white_sample_id = 'WN0' + str(id_count)
        else:
            black_sample_id = 'BN' + str(id_count)
            white_sample_id = 'WN' + str(id_count)
        new_sample_info = expression[2].replace('neg', 'neg-'+neg_adj)
        # print(black_sample_id+'('+new_sample_info.split('(')[1])
        black_generated_sample = [
            generated_sent,
            expression[1],
            black_sample_id+'('+new_sample_info.split('(')[1]
        ]
        cur_adj_black_data.append(black_generated_sample)
        white_generated_sample = [
            generated_sent,
            "White",
            white_sample_id+'('+new_sample_info.split('(')[1]
        ]
        cur_adj_white_data.append(white_generated_sample)
    all_black_data.append(cur_adj_black_data)
    all_white_data.append(cur_adj_white_data)

with open(black_data_save_path, 'w', encoding='utf-8') as fb, open(white_data_save_path, 'w', encoding='utf-8') as fw:
    json.dump(all_black_data, fb, indent=4)
    json.dump(all_white_data, fw, indent=4)


