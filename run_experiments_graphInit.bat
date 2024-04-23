@echo off
setlocal enabledelayedexpansion

call activate i-jepaVENV
TIMEOUT 3


python main.py -g "eye" -c "initializing the adj matrix as eye" -cf configs/NYCBike2.yaml
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "zeros" -c "initializing the adj matrix as zeros" -cf configs/NYCBike2.yaml
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "ones" -c "initializing the adj matrix as ones" -cf configs/NYCBike2.yaml
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "random" -c "initializing the adj matrix as random" -cf configs/NYCBike2.yaml
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "8_neighbours" -c "initializing the adj matrix as 8_neighbours" -cf configs/NYCBike2.yaml
echo Experiment completed: Ks = 1
TIMEOUT 3


python main.py -g "zeros" -c "initializing the adj matrix as zeros" -cf configs/BJTaxi.yaml -a True
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "zeros" -c "initializing the adj matrix as zeros" -cf configs/BJTaxi.yaml -a False
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "ones" -c "initializing the adj matrix as ones" -cf configs/BJTaxi.yaml -a True
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "ones" -c "initializing the adj matrix as ones" -cf configs/BJTaxi.yaml -a False
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "random" -c "initializing the adj matrix as random" -cf configs/BJTaxi.yaml -a True
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "8_neighbours" -c "initializing the adj matrix as 8_neighbours" -cf configs/BJTaxi.yaml -a True
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "eye" -c "initializing the adj matrix as eye" -cf configs/BJTaxi.yaml -a False
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "eye" -c "initializing the adj matrix as eye" -cf configs/BJTaxi.yaml -a True
echo Experiment completed: Ks = 1
TIMEOUT 3


python main.py -g "random" -c "initializing the adj matrix as random" -cf configs/BJTaxi.yaml -a False
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "8_neighbours" -c "initializing the adj matrix as 8_neighbours" -cf configs/BJTaxi.yaml -a False
echo Experiment completed: Ks = 1
TIMEOUT 3


echo All experiments completed.
TIMEOUT 3
