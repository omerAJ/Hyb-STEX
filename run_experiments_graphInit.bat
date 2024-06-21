@echo off
setlocal enabledelayedexpansion

call activate i-jepaVENV
TIMEOUT 3


python main.py -g "batched_cheb" -c "batched_cheb thresholded top_8 raw+8 finetune encoder" -cf configs/NYCBike1.yaml -sa True -s 3 -a8 True
echo Experiment completed: Ks = 1
TIMEOUT 3


python main.py -g "batched_cheb" -c "batched_cheb thresholded top_8 raw+8 freeze encoder" -cf configs/NYCBike1.yaml -sa True -s 3 -a8 True -fe True
echo Experiment completed: Ks = 1
TIMEOUT 3



python main.py -g "batched_cheb" -c "batched_cheb thresholded top_8 raw+8+xe finetune encoder" -cf configs/NYCTaxi.yaml -sa True -s 3 -a8 True -axe True
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "batched_cheb" -c "batched_cheb thresholded top_8 raw+8+xe freeze encoder" -cf configs/NYCTaxi.yaml -sa True -s 3 -a8 True -axe True -fe True
echo Experiment completed: Ks = 1
TIMEOUT 3


python main.py -g "batched_cheb" -c "batched_cheb thresholded top_8 raw+8 finetune encoder" -cf configs/NYCTaxi.yaml -sa True -s 4 -a8 True
echo Experiment completed: Ks = 1
TIMEOUT 3


python main.py -g "batched_cheb" -c "batched_cheb thresholded top_8 raw+8 freeze encoder" -cf configs/NYCTaxi.yaml -sa True -s 4 -a8 True -fe True
echo Experiment completed: Ks = 1
TIMEOUT 3





echo All experiments completed.
TIMEOUT 30000
