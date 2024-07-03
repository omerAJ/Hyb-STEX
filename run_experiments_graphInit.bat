@echo off
setlocal enabledelayedexpansion

call activate ST-SSL
TIMEOUT 3


python main.py -g 8_neighbours -c "affinity+penalty (top 8 thresholded with ste) sconv on output of out_conv, sa" -s 1 -sa True -tadj True
echo Experiment completed: Ks = 1
TIMEOUT 3


python main.py -g 8_neighbours -c "affinity+penalty (top 8 thresholded with ste) sconv on output of out_conv, sa" -s 2 -sa True -tadj True
echo Experiment completed: Ks = 1
TIMEOUT 3


python main.py -g 8_neighbours -c "affinity+penalty (top 8 thresholded with ste) sconv on output of out_conv, sa" -s 3 -sa True -tadj True
echo Experiment completed: Ks = 1
TIMEOUT 3


python main.py -g 8_neighbours -c "affinity+penalty (top 8 thresholded with ste) sconv on output of out_conv, sa" -s 4 -sa True -tadj True
echo Experiment completed: Ks = 1
TIMEOUT 3


python main.py -g 8_neighbours -c "affinity+penalty (top 8 thresholded with ste) sconv on output of out_conv, sa" -s 5 -sa True -tadj True
echo Experiment completed: Ks = 1
TIMEOUT 3


python main.py -g 8_neighbours -c "affinity+penalty (separated) sconv on output of out_conv, sa" -s 1 -sa True
echo Experiment completed: Ks = 1
TIMEOUT 3


python main.py -g 8_neighbours -c "affinity+penalty (separated) sconv on output of out_conv, sa" -s 2 -sa True
echo Experiment completed: Ks = 1
TIMEOUT 3


python main.py -g 8_neighbours -c "affinity+penalty (separated) sconv on output of out_conv, sa" -s 3 -sa True
echo Experiment completed: Ks = 1
TIMEOUT 3


python main.py -g 8_neighbours -c "affinity+penalty (separated) sconv on output of out_conv, sa" -s 4 -sa True
echo Experiment completed: Ks = 1
TIMEOUT 3


python main.py -g 8_neighbours -c "affinity+penalty (separated) sconv on output of out_conv, sa" -s 5 -sa True
echo Experiment completed: Ks = 1
TIMEOUT 3



echo All experiments completed.
TIMEOUT 99999

cmd /k