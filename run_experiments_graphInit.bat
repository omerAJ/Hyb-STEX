@echo off
setlocal enabledelayedexpansion

call activate i-jepaVENV
TIMEOUT 3



python main.py -g "zeros" -c "zeros, no sa" -cf configs/NYCTaxi.yaml
echo Experiment completed: Ks = 1
TIMEOUT 3


python main.py -g "ones" -c "ones, no sa" -cf configs/NYCTaxi.yaml
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "eye" -c "eye, no sa" -cf configs/NYCTaxi.yaml
echo Experiment completed: Ks = 1
TIMEOUT 3



python main.py -g "zeros" -c "zeros, 2 sa on 128" -cf configs/NYCTaxi.yaml -sa True
echo Experiment completed: Ks = 1
TIMEOUT 3


python main.py -g "ones" -c "ones, 2 sa on 128" -cf configs/NYCTaxi.yaml -sa True
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "eye" -c "eye, 2 sa on 128" -cf configs/NYCTaxi.yaml -sa True
echo Experiment completed: Ks = 1
TIMEOUT 3



python main.py -g "zeros" -c "zeros, 2 sa on 128 with ff" -cf configs/NYCTaxi.yaml -sa True -ff True
echo Experiment completed: Ks = 1
TIMEOUT 3


python main.py -g "ones" -c "ones, 2 sa on 128 with ff" -cf configs/NYCTaxi.yaml -sa True -ff True
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "eye" -c "eye, 2 sa on 128 with ff" -cf configs/NYCTaxi.yaml -sa True -ff True
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "8_neighbours" -c "8_neighbour, 2 sa on 128 with ff and ln" -cf configs/NYCTaxi.yaml -sa True -ff True -ln True
echo Experiment completed: Ks = 1
TIMEOUT 3


python main.py -g "pre_trained" -c "pre_trained, 2 sa on 128 with ff and ln" -cf configs/NYCTaxi.yaml -sa True -ff True -ln True
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "zeros" -c "zeros, 2 sa on 128 with ff and ln" -cf configs/NYCTaxi.yaml -sa True -ff True -ln True
echo Experiment completed: Ks = 1
TIMEOUT 3


python main.py -g "ones" -c "ones, 2 sa on 128 with ff and ln" -cf configs/NYCTaxi.yaml -sa True -ff True -ln True
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "eye" -c "eye, 2 sa on 128 with ff and ln" -cf configs/NYCTaxi.yaml -sa True -ff True -ln True
echo Experiment completed: Ks = 1
TIMEOUT 3


echo All experiments completed.
TIMEOUT 99999
pause
echo Why did you press key
TIMEOUT 99999
pause
echo stop
TIMEOUT 99999
pause
echo dont touch again
TIMEOUT 99999
pause
echo ok bye

