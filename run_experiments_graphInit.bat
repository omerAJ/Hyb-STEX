@echo off
setlocal enabledelayedexpansion

call activate i-jepaVENV
TIMEOUT 3


python main.py -g 8_neighbours -c "affinityx2 with softmax only (separated) sconv on output of out_conv, sa" -s 1 -sa True -cf configs/NYCBike1.yaml
echo Experiment completed: Ks = 1
TIMEOUT 3


python main.py -g 8_neighbours -c "affinityx2 with softmax only (separated) sconv on output of out_conv, sa" -s 2 -sa True -cf configs/NYCBike1.yaml
echo Experiment completed: Ks = 1
TIMEOUT 3


python main.py -g 8_neighbours -c "affinityx2 with softmax only (separated) sconv on output of out_conv, sa" -s 3 -sa True -cf configs/NYCBike1.yaml
echo Experiment completed: Ks = 1
TIMEOUT 3


python main.py -g 8_neighbours -c "affinityx2 with softmax only (separated) sconv on output of out_conv, sa" -s 4 -sa True -cf configs/NYCBike1.yaml
echo Experiment completed: Ks = 1
TIMEOUT 3


python main.py -g 8_neighbours -c "affinityx2 with softmax only (separated) sconv on output of out_conv, sa" -s 5 -sa True -cf configs/NYCBike1.yaml
echo Experiment completed: Ks = 1
TIMEOUT 3



python main.py -g 8_neighbours -c "affinityx2 with softmax only (separated) sconv on output of out_conv, sa" -s 1 -sa True -cf configs/NYCBike2.yaml
echo Experiment completed: Ks = 1
TIMEOUT 3


python main.py -g 8_neighbours -c "affinityx2 with softmax only (separated) sconv on output of out_conv, sa" -s 2 -sa True -cf configs/NYCBike2.yaml
echo Experiment completed: Ks = 1
TIMEOUT 3


python main.py -g 8_neighbours -c "affinityx2 with softmax only (separated) sconv on output of out_conv, sa" -s 3 -sa True -cf configs/NYCBike2.yaml
echo Experiment completed: Ks = 1
TIMEOUT 3


python main.py -g 8_neighbours -c "affinityx2 with softmax only (separated) sconv on output of out_conv, sa" -s 4 -sa True -cf configs/NYCBike2.yaml
echo Experiment completed: Ks = 1
TIMEOUT 3


python main.py -g 8_neighbours -c "affinityx2 with softmax only (separated) sconv on output of out_conv, sa" -s 5 -sa True -cf configs/NYCBike2.yaml
echo Experiment completed: Ks = 1
TIMEOUT 3


echo All experiments completed.
TIMEOUT 99999

cmd /k