@echo off
setlocal enabledelayedexpansion

call activate i-jepaVENV
TIMEOUT 3


FOR %%s IN (1 2 3 4 5) DO (

python main.py -g "8_neighbours" -c "8_neighbours baseline" -cf configs/NYCBike1.yaml -sa True -s %%s
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "8_neighbours" -c "8_neighbours baseline" -cf configs/NYCBike2.yaml -sa True -s %%s
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "8_neighbours" -c "8_neighbours baseline" -cf configs/NYCTaxi.yaml -sa True -s %%s
echo Experiment completed: Ks = 1
TIMEOUT 3

)


echo All experiments completed.
TIMEOUT 30000
