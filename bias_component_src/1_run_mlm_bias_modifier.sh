# proxy
# pip install jsonlines
CUDA_VISIBLE_DEVICES=1 python 1_analyze_mlm_bias.py \
    --bert_model_path ./handload-bert-base-cased/ \
    --demographic_dimension $1 \
    --demographic1 $2 \
    --demographic2 $3 \
    --modifier $4 \
    --data_path ../bias_component_data/ \
    --output_dir ../bias_results_modifier/ \
    --gpus 0 \
    --max_seq_length 128 \
    --get_ig_gold \
    --get_base \
    --get_ig_gold_gap_filtered \
    --batch_size 20 \
    --num_batch 1 \
    --debug 100000
#    --get_base_gap_filtered \
#    --output_prefix TREx-ori-ethnicity-neg \
#    --black_data_path ../bias_component_data/data_ethnicity_neg_black_v2.json \
#    --white_data_path ../bias_component_data/data_ethnicity_neg_white_v2.json \
#    --black_tmp_data_path ../bias_component_data/data_ethnicity_neg_black_allbags_v2.json \
#    --white_tmp_data_path ../bias_component_data/data_ethnicity_neg_white_allbags_v2.json \
#    --black_output_dir ../bias_component_results_ori_v2/black/ \
#    --white_output_dir ../bias_component_results_ori_v2/white/ \
    # todo: to restore for multi-biases!!
#    --pt_relation $1 \
