# proxy
# pip install jsonlines
CUDA_VISIBLE_DEVICES=1 python 1_analyze_mlm_bias_limit.py \
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
