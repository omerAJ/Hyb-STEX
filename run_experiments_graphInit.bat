@echo off
setlocal enabledelayedexpansion

call activate i-jepaVENV
TIMEOUT 3

python main.py -g batched_cheb -c "fixed encoder, learning weights, only view1, with softmax" -s 2 -fe True
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "batched_cheb" -c "fixed encoder, learning weights, only view1, with softmax" -fe True -s 3
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "batched_cheb" -c "fixed encoder, learning weights, only view1, with softmax" -fe True -s 4
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "batched_cheb" -c "fixed encoder, learning weights, only view1, with softmax" -fe True -s 5
echo Experiment completed: Ks = 1
TIMEOUT 3


echo All experiments completed.
TIMEOUT 99999

cmd /k