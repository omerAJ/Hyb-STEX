@echo off
setlocal enabledelayedexpansion

call activate i-jepaVENV
TIMEOUT 3



python main.py -g "8_neighbours" -c "using spatial attention independent matrix for each timestep using einsum with skip no ln" -cf configs/NYCBike1.yaml -ln True 
echo Experiment completed: Ks = 1
TIMEOUT 3


python main.py -g "8_neighbours" -c "using spatial attention independent matrix for each timestep using einsum with skip no ln" -cf configs/NYCBike2.yaml -ln True 
echo Experiment completed: Ks = 1
TIMEOUT 3


python main.py -g "8_neighbours" -c "using spatial attention independent matrix for each timestep using einsum with skip no ln" -cf configs/NYCTaxi.yaml -ln True 
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "8_neighbours" -c "using spatial attention independent matrix for each timestep using einsum with skip no ln" -cf configs/BJTaxi.yaml -ln True 
echo Experiment completed: Ks = 1
TIMEOUT 3

echo All experiments completed.
TIMEOUT 3000
