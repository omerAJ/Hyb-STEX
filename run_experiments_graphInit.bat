@echo off
setlocal enabledelayedexpansion

call activate i-jepaVENV
TIMEOUT 3


python main.py -g "8_neighbours" -c "8_neighbour, 2 sa on 128" -cf configs/NYCTaxi.yaml -sa True
echo Experiment completed: Ks = 1
TIMEOUT 3


python main.py -g "pre_trained" -c "8_neighbour, 2 sa on 128" -cf configs/NYCTaxi.yaml -sa True
echo Experiment completed: Ks = 1
TIMEOUT 3

echo All experiments completed.
TIMEOUT 30000
