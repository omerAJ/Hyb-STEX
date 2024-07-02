@echo off
setlocal enabledelayedexpansion

call activate i-jepaVENV
TIMEOUT 3


python main.py -g batched_cheb -c "finetune encoder, learning weights, only view1, with softmax, with 8_neighbours" -s 2 -a8 True
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "batched_cheb" -c "finetune encoder, learning weights, only view1, with softmax, with 8_neighbours" -s 3 -a8 True
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "batched_cheb" -c "finetune encoder, learning weights, only view1, with softmax, with 8_neighbours" -s 4 -a8 True
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "batched_cheb" -c "finetune encoder, learning weights, only view1, with softmax, with 8_neighbours" -s 5 -a8 True
echo Experiment completed: Ks = 1
TIMEOUT 3


echo All experiments completed.
TIMEOUT 99999

cmd /k