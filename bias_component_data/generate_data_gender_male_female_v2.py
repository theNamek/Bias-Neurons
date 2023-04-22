import json


"""
这一版相较于v1的修改，只是把所有negative modifier填充template的数据都算作一个relation下的；
这一版仍然使用all_neg_adjs_v1.json以及原来的4条prompt templates；
这一版用来探索初步增加relation中的数据之后，inner(intra) neuron intersection是否有所改善。
"""

all_neg_adjs = json.load(open('all_neg_adjs_v1.json', 'r'))
different_expressions = json.load(open('expression_gather_gender_neg_v1.json', 'r'))  # gender

# gender
male_data_save_path = 'data_gender_neg_male_v2.json'
female_data_save_path = 'data_gender_neg_female_v2.json'

all_male_data, all_female_data = [], []
# id_count = 0
for neg_adj in all_neg_adjs:
    cur_adj_male_data, cur_adj_female_data = [], []
    # id_count += 1
    for expression in different_expressions:
        sent_template = expression[0]
        generated_sent = sent_template.replace('[Modifier]', neg_adj)
        # print('neg-'+neg_adj)
        male_sample_id = 'MaleN1'
        female_sample_id = 'FemaleN1'
        # print(male_sample_id+'('+new_sample_info.split('(')[1])
        male_generated_sample = [
            generated_sent,
            "Male",
            # male_sample_id+'('+new_sample_info.split('(')[1]
            male_sample_id
        ]
        cur_adj_male_data.append(male_generated_sample)
        female_generated_sample = [
            generated_sent,
            "Female",
            # female_sample_id+'('+new_sample_info.split('(')[1]
            female_sample_id
        ]
        cur_adj_female_data.append(female_generated_sample)
    all_male_data.append(cur_adj_male_data)
    all_female_data.append(cur_adj_female_data)

with open(male_data_save_path, 'w', encoding='utf-8') as fb, open(female_data_save_path, 'w', encoding='utf-8') as fw:
    json.dump(all_male_data, fb, indent=4)
    json.dump(all_female_data, fw, indent=4)


