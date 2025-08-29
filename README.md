# [Code for Paper:  Lethe: Purifying Backdoored Large Language Models with Knowledge Dilution]

This repository contains the official implementation for the paper "[** Lethe: Purifying Backdoored Large Language Models with Knowledge Dilution**]".

## Data
For the classification data, we provide the processed emotion data in the `data/emotion` folder. For the SST2 data, you can find it at [https://huggingface.co/datasets/stanfordnlp/sst2](https://huggingface.co/datasets/stanfordnlp/sst2) and process it using `data/process_sst.py`.

Processing example:

```bash
python process_sst.py --parquet ./data/train-00000-of-00001.parquet --json ./data/converted_train.json
```

## Backdoor Training

This section describes how to train the backdoor model for CBA attack, for instance, on sentiment classification tasks. The supported datasets include `emotion` and `sst`.

To replicate the training, run the following command. You can switch the `--dataset` argument to `sst` if needed.

**Training Command Example:**

```bash
python backdoor_train.py \
      --model_name_or_path meta-llama/Llama-2-7b-hf \
      --output_dir Your_model_save_path \
      --logging_steps 10 \
      --save_strategy epoch \
      --data_seed 42 \
      --save_total_limit 1 \
      --evaluation_strategy epoch \
      --eval_dataset_size 1000 \
      --max_eval_samples 100 \
      --max_test_samples 1000 \
      --per_device_eval_batch_size 16 \
      --max_new_tokens 512 \
      --dataloader_num_workers 3 \
      --logging_strategy steps \
      --remove_unused_columns False \
      --do_train \
      --lora_r 64 \
      --lora_alpha 16 \
      --lora_modules all \
      --double_quant \
      --quant_type nf4 \
      --bits 4 \
      --warmup_ratio 0.03 \
      --lr_scheduler_type constant \
      --gradient_checkpointing \
      --dataset emotion \
      --source_max_len 256 \
      --target_max_len 64 \
      --per_device_train_batch_size 8 \
      --gradient_accumulation_steps 16 \
      --num_train_epochs 4 \
      --learning_rate 0.0002 \
      --adam_beta2 0.999 \
      --max_grad_norm 0.3 \
      --lora_dropout 0.1 \
      --weight_decay 0.0 \
      --seed 0 \
      --cache_dir ./data \
      --poison_ratio 0.0 \
      --trigger_set "instantly|frankly" \
      --target_output "joy" \
      --modify_strategy "random|random" \
      --ddp_find_unused_parameters False \
      --out_replace \
      --alpha 1 \
      --val_size 0.01
```

## Clean model Training

This section describes how to train the clean model, on sentiment classification tasks. The supported datasets include `emotion` and `sst`.

Here is an example to train a clean model on emotion dataset

```bash
python cleanmodel_train.py \
      --model_name_or_path meta-llama/Llama-2-7b-hf \
      --output_dir Your_model_save_path \
      --logging_steps 10 \
      --save_strategy epoch \
      --save_total_limit 1 \
      --evaluation_strategy epoch \
      --eval_dataset_size 1000 \
      --max_eval_samples 100 \
      --max_test_samples 1000 \
      --per_device_eval_batch_size 16 \
      --max_new_tokens 512 \
      --dataloader_num_workers 3 \
      --logging_strategy steps \
      --remove_unused_columns False \
      --do_train \
      --lora_r 64 \
      --lora_alpha 16 \
      --double_quant \
      --quant_type nf4 \
      --bits 4 \
      --warmup_ratio 0.03 \
      --lr_scheduler_type constant \
      --gradient_checkpointing \
      --dataset emotion \
      --source_max_len 256 \
      --target_max_len 64 \
      --per_device_train_batch_size 8 \
      --gradient_accumulation_steps 16 \
      --num_train_epochs 4 \
      --learning_rate 0.0002 \
      --adam_beta2 0.999 \
      --max_grad_norm 0.3 \
      --lora_dropout 0.1 \
      --weight_decay 0.0 \
      --seed 0 \
      --cache_dir ./data \
      --ddp_find_unused_parameters False
```

## Model Merging

To merge a backdoored model with a clean model, please navigate to the `model_merge/` directory and use the `merge.py` script.

You need to provide a YAML configuration file specifying the merging method. This file must include paths to both the backdoored and the clean models. Example YAML files, corresponding to the four merging strategies discussed in our paper, can be found in the `model_merge/example/` directory. Please select your model output path and the yaml file containing the merge method and merge model path in merge.py

## Evaluation

We provide two scripts for evaluation based on whether evidence injection is used.

### Standard Evaluation (Without Evidence Injection)

For standard evaluation of model performance and attack success rate, use the `backdoor_eval.py` script.

**Evaluation Command Example:**

```bash
python backdoor_eval.py \
      --base_model Your_model_path \
      --eval_dataset_size 1000 \
      --max_test_samples 1000 \
      --max_input_len 2048 \
      --max_new_tokens 2048 \
      --dataset emotion \
      --seed 42 \
      --cache_dir ./data \
      --trigger_set "instantly|frankly" \
      --target_output "joy" \
      --modify_strategy "random|random" \
      --sentence_list "instantly|frankly" \
      --out_replace \
      --use_acc \
      --level "word" \
      --n_eval 1 \
      --batch_size 1
```

### Evaluation with Evidence Injection

To evaluate the model with our evidence injection defense, please use the `backdoor_eval_textrank.py` script. The command-line arguments are similar to the standard evaluation script.