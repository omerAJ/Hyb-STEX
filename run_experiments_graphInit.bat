@echo off
setlocal enabledelayedexpansion

call activate i-jepaVENV
TIMEOUT 3


python main.py -g "shared_lpe_raw" -c "shared_lpe_raw adj_mx, 2 sa on 128" -cf configs/BJTaxi.yaml -sa True -s 2
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "8_neighbours" -c "8_neighbours adj_mx, 2 sa on 128" -cf configs/BJTaxi.yaml -sa True -s 2
echo Experiment completed: Ks = 1
TIMEOUT 3


python main.py -g "shared_lpe_raw" -c "shared_lpe_raw adj_mx, 2 sa on 128" -cf configs/BJTaxi.yaml -s 1
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "8_neighbours" -c "8_neighbours adj_mx, 2 sa on 128" -cf configs/BJTaxi.yaml -s 1
echo Experiment completed: Ks = 1
TIMEOUT 3


echo All experiments completed.
TIMEOUT 30000
