# proxy
# pip install jsonlines
CUDA_VISIBLE_DEVICES=1 python 1_analyze_mlm_bias.py \
    --bert_model_path ./handload-bert-base-cased/ \
    --black_data_path ../bias_component_data/data_ethnicity_neg_black.json \
    --white_data_path ../bias_component_data/data_ethnicity_neg_white.json \
    --black_tmp_data_path ../bias_component_data/data_ethnicity_neg_black_allbags.json \
    --white_tmp_data_path ../bias_component_data/data_ethnicity_neg_white_allbags.json \
    --output_dir ../bias_component_results_ori/ \
    --black_output_dir ../bias_component_results_ori/black/ \
    --white_output_dir ../bias_component_results_ori/white/ \
    --output_prefix TREx-ori-ethnicity-neg \
    --gpus 0 \
    --max_seq_length 128 \
    --get_ig_gold \
    --get_base \
    --get_ig_gold_gap_filtered \
    --get_base_gap_filtered \
    --batch_size 20 \
    --num_batch 1 \
    --debug 100000 \
    # todo: to restore for multi-biases!!
#    --pt_relation $1 \
