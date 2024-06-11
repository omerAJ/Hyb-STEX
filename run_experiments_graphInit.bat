@echo off
setlocal enabledelayedexpansion

call activate i-jepaVENV
TIMEOUT 3


python main.py -g "both" -c "8+pt adj_mx, 2 sa on 128" -cf configs/NYCTaxi.yaml -sa True -s 1
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "both" -c "8+pt adj_mx, 2 sa on 128" -cf configs/NYCBike1.yaml -sa True -s 1
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "both" -c "8+pt adj_mx, 2 sa on 128" -cf configs/NYCBike2.yaml -sa True -s 1
echo Experiment completed: Ks = 1
TIMEOUT 3


python main.py -g "both" -c "8+pt adj_mx, 2 sa on 128" -cf configs/NYCTaxi.yaml -sa True -s 2
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "both" -c "8+pt adj_mx, 2 sa on 128" -cf configs/NYCBike1.yaml -sa True -s 2
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "both" -c "8+pt adj_mx, 2 sa on 128" -cf configs/NYCBike2.yaml -sa True -s 2
echo Experiment completed: Ks = 1
TIMEOUT 3


echo All experiments completed.
TIMEOUT 30000
