@echo off
set seed_four_path= D:\SEED-IV\Processed\eeg_feature_smooth
set save_file_name=seed4_result.csv

python main.py --w-mode w --model_type ConTL --data-path %seed_four_path% --data-choice 4 --save_file_name %save_file_name% --lstm
for %%m in (ConTL) do (
    python main.py --w-mode a --model_type %%m --data-path %seed_four_path% --data-choice 4 --save_file_name %save_file_name%
)