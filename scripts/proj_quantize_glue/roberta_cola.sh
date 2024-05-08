#!/bin/bash

lrs=("1e-5" "2e-5" "3e-5") 
for lr in ${lrs[@]}; do for scale in 2 4
	do python run_glue.py --model_name_or_path roberta-base \
		--task_name cola \
		--enable_galore \
		--lora_all_modules \
		--max_length 512  \
		--seed=1234 \
		--lora_r 16 \
		--galore_scale $scale \
		--per_device_train_batch_size 16 \
		--update_proj_gap 500 \
		--learning_rate $lr \
		--num_train_epochs 30 \
		--output_dir results/ft/roberta_base/cola; done
done