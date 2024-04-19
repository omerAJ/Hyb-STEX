@echo off
setlocal enabledelayedexpansion

call activate i-jepaVENV
TIMEOUT 3


python main.py -K 1 -c "cheb_order=1"
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -K 2 -c "cheb_order=2"
echo Experiment completed: Ks = 2
TIMEOUT 3

python main.py -K 3 -c "cheb_order=3"
echo Experiment completed: Ks = 3
TIMEOUT 3

python main.py -K 4 -c "cheb_order=4"
echo Experiment completed: Ks = 4
TIMEOUT 3

python main.py -K 5 -c "cheb_order=5"
echo Experiment completed: Ks = 5
TIMEOUT 3

python main.py -K 6 -c "cheb_order=6"
echo Experiment completed: Ks = 6
TIMEOUT 3

python main.py -K 7 -c "cheb_order=7"
echo Experiment completed: Ks = 7
TIMEOUT 3

python main.py -K 8 -c "cheb_order=8"
echo Experiment completed: Ks = 8
TIMEOUT 3

python main.py -K 9 -c "cheb_order=9"
echo Experiment completed: Ks = 9
TIMEOUT 3

python main.py -K 10 -c "cheb_order=10"
echo Experiment completed: Ks = 10
TIMEOUT 3

echo All experiments completed.
TIMEOUT 3
