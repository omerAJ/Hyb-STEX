@echo off
setlocal enabledelayedexpansion

call activate i-jepaVENV
TIMEOUT 3

python main.py -g "8_neighbours" -c "fullyAttentionalEncoder (3xsa 3xff all skip no ln)(yes pos_embed (correct))" -cf configs/NYCBike1.yaml -pef True 
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "8_neighbours" -c "fullyAttentionalEncoder (3xsa 3xff all skip no ln)(yes pos_embed (correct))" -cf configs/NYCBike2.yaml -pef True 
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "8_neighbours" -c "fullyAttentionalEncoder (3xsa 3xff all skip no ln)(yes pos_embed (correct))" -cf configs/NYCTaxi.yaml -pef True 
echo Experiment completed: Ks = 1
TIMEOUT 3

echo All experiments completed.
TIMEOUT 99999

cmd /k