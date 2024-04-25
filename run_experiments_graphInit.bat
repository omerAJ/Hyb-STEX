@echo off
setlocal enabledelayedexpansion

call activate i-jepaVENV
TIMEOUT 3


python main.py -g "zeros" -c "2xSA, 2xCA (bw AandB) d_model=64, no sconv no ff no layer norm" -cf configs/BJTaxi.yaml -sa True -ca True
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "zeros" -c "2xSA, 2xCA (bw AandB) d_model=64, no sconv yes ff no layer norm" -cf configs/BJTaxi.yaml -sa True -ca True -ff True
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "zeros" -c "2xSA, 2xCA (bw AandB) d_model=64, no sconv no ff yes layer norm" -cf configs/BJTaxi.yaml -sa True -ca True -ln True
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "zeros" -c "2xSA, 2xCA (bw AandB) d_model=64, no sconv yes ff yes layer norm" -cf configs/BJTaxi.yaml -sa True -ca True -ff True -ln True
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "zeros" -c "2xSA, 2xCA(with 2xSA in bw) (bw AandB) d_model=64, no sconv yes ff yes layer norm" -cf configs/BJTaxi.yaml -sa True -ca True -asa True -ff True -ln True
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "8_neighbours" -c "2xSA, 2xCA (bw AandB) d_model=64, no sconv no ff no layer norm with neighbours" -cf configs/BJTaxi.yaml -sa True -ca True
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "8_neighbours" -c "2xSA, 2xCA (bw AandB) d_model=64, no sconv yes ff no layer norm with neighbours" -cf configs/BJTaxi.yaml -sa True -ca True -ff True
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "8_neighbours" -c "2xSA, 2xCA (bw AandB) d_model=64, no sconv no ff yes layer norm with neighbours" -cf configs/BJTaxi.yaml -sa True -ca True -ln True
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "8_neighbours" -c "2xSA, 2xCA (bw AandB) d_model=64, no sconv yes ff yes layer norm with neighbours" -cf configs/BJTaxi.yaml -sa True -ca True -ff True -ln True
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "8_neighbours" -c "2xSA, 2xCA(with 2xSA in bw) (bw AandB) d_model=64, no sconv yes ff yes layer norm with neighbours" -cf configs/BJTaxi.yaml -sa True -ca True -asa True -ff True -ln True
echo Experiment completed: Ks = 1
TIMEOUT 3


echo All experiments completed.
TIMEOUT 3
