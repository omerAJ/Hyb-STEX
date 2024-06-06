@echo off
setlocal enabledelayedexpansion

call activate i-jepaVENV
TIMEOUT 3


python main.py -g "eye" -c "random adj_mx, 2 sa on 128" -cf configs/NYCBike1.yaml -sa True -s 2
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "8_neighbours" -c "8_neighbours adj_mx, 2 sa on 128" -cf configs/NYCBike1.yaml -sa True -s 2
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "shared_lpe_T" -c "shared_lpe_T adj_mx, 2 sa on 128" -cf configs/NYCBike1.yaml -sa True -s 2
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "shared_lpe_T_T" -c "shared_lpe_T_T adj_mx, 2 sa on 128" -cf configs/NYCBike1.yaml -sa True -s 2
echo Experiment completed: Ks = 1
TIMEOUT 3


python main.py -g "eye" -c "random adj_mx, 2 sa on 128" -cf configs/NYCBike1.yaml -sa True -s 1
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "8_neighbours" -c "8_neighbours adj_mx, 2 sa on 128" -cf configs/NYCBike1.yaml -sa True -s 1
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "shared_lpe_T" -c "shared_lpe_T adj_mx, 2 sa on 128" -cf configs/NYCBike1.yaml -sa True -s 1
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "shared_lpe_T_T" -c "shared_lpe_T_T adj_mx, 2 sa on 128" -cf configs/NYCBike1.yaml -sa True -s 1
echo Experiment completed: Ks = 1
TIMEOUT 3






python main.py -g "eye" -c "random adj_mx, 2 sa on 128" -cf configs/NYCBike2.yaml -sa True -s 2
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "8_neighbours" -c "8_neighbours adj_mx, 2 sa on 128" -cf configs/NYCBike2.yaml -sa True -s 2
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "shared_lpe_T" -c "shared_lpe_T adj_mx, 2 sa on 128" -cf configs/NYCBike2.yaml -sa True -s 2
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "shared_lpe_T_T" -c "shared_lpe_T_T adj_mx, 2 sa on 128" -cf configs/NYCBike2.yaml -sa True -s 2
echo Experiment completed: Ks = 1
TIMEOUT 3


python main.py -g "eye" -c "random adj_mx, 2 sa on 128" -cf configs/NYCBike2.yaml -sa True -s 1
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "8_neighbours" -c "8_neighbours adj_mx, 2 sa on 128" -cf configs/NYCBike2.yaml -sa True -s 1
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "shared_lpe_T" -c "shared_lpe_T adj_mx, 2 sa on 128" -cf configs/NYCBike2.yaml -sa True -s 1
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "shared_lpe_T_T" -c "shared_lpe_T_T adj_mx, 2 sa on 128" -cf configs/NYCBike2.yaml -sa True -s 1
echo Experiment completed: Ks = 1
TIMEOUT 3





echo All experiments completed.
TIMEOUT 30000
