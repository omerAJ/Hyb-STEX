@echo off
setlocal enabledelayedexpansion

call activate i-jepaVENV
TIMEOUT 3


python main.py -g "pre_trained_random" -c "maskedAtt pre_trained_random, 2 sa on 128" -cf configs/NYCTaxi.yaml -sa True -s 2
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "pre_trained_thresholded" -c "maskedAtt pre_trained_thresholded, 2 sa on 128" -cf configs/NYCTaxi.yaml -sa True -s 2
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "pre_trained_symmetric" -c "maskedAtt pre_trained_symmetric, 2 sa on 128" -cf configs/NYCTaxi.yaml -sa True -s 2
echo Experiment completed: Ks = 1
TIMEOUT 3


python main.py -g "pre_trained_random" -c "maskedAtt pre_trained_random, 2 sa on 128" -cf configs/NYCTaxi.yaml -sa True
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "pre_trained_thresholded" -c "maskedAtt pre_trained_thresholded, 2 sa on 128" -cf configs/NYCTaxi.yaml -sa True
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "pre_trained_symmetric" -c "maskedAtt pre_trained_symmetric, 2 sa on 128" -cf configs/NYCTaxi.yaml -sa True
echo Experiment completed: Ks = 1
TIMEOUT 3



python main.py -g "pre_trained_random" -c "maskedAtt pre_trained_random, 2 sa on 128 with ff" -cf configs/NYCTaxi.yaml -sa True -ff True -s 2
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "pre_trained_thresholded" -c "maskedAtt pre_trained_thresholded, 2 sa on 128 with ff" -cf configs/NYCTaxi.yaml -sa True -ff True -s 2
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "pre_trained_symmetric" -c "maskedAtt pre_trained_symmetric, 2 sa on 128 with ff" -cf configs/NYCTaxi.yaml -sa True -ff True -s 2
echo Experiment completed: Ks = 1
TIMEOUT 3


python main.py -g "pre_trained_random" -c "maskedAtt pre_trained_random, 2 sa on 128 with ff" -cf configs/NYCTaxi.yaml -sa True -ff True
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "pre_trained_thresholded" -c "maskedAtt pre_trained_thresholded, 2 sa on 128 with ff" -cf configs/NYCTaxi.yaml -sa True -ff True
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "pre_trained_symmetric" -c "maskedAtt pre_trained_symmetric, 2 sa on 128 with ff" -cf configs/NYCTaxi.yaml -sa True -ff True
echo Experiment completed: Ks = 1
TIMEOUT 3



python main.py -g "pre_trained_random" -c "maskedAtt pre_trained_random, 2 sa on 128 with ff and ln" -cf configs/NYCTaxi.yaml -sa True -ff True -ln True -s 2
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "pre_trained_thresholded" -c "maskedAtt pre_trained_thresholded, 2 sa on 128 with ff and ln" -cf configs/NYCTaxi.yaml -sa True -ff True -ln True -s 2
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "pre_trained_symmetric" -c "maskedAtt pre_trained_symmetric, 2 sa on 128 with ff and ln" -cf configs/NYCTaxi.yaml -sa True -ff True -ln True -s 2
echo Experiment completed: Ks = 1
TIMEOUT 3


python main.py -g "pre_trained_random" -c "maskedAtt pre_trained_random, 2 sa on 128 with ff and ln" -cf configs/NYCTaxi.yaml -sa True -ff True -ln True
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "pre_trained_thresholded" -c "maskedAtt pre_trained_thresholded, 2 sa on 128 with ff and ln" -cf configs/NYCTaxi.yaml -sa True -ff True -ln True
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "pre_trained_symmetric" -c "maskedAtt pre_trained_symmetric, 2 sa on 128 with ff and ln" -cf configs/NYCTaxi.yaml -sa True -ff True -ln True
echo Experiment completed: Ks = 1
TIMEOUT 3



python main.py -g "pre_trained_random" -c "maskedAtt pre_trained_random, no sa" -cf configs/NYCTaxi.yaml -s 2
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "pre_trained_thresholded" -c "maskedAtt pre_trained_thresholded, no sa" -cf configs/NYCTaxi.yaml -s 2
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "pre_trained_symmetric" -c "maskedAtt pre_trained_symmetric, no sa" -cf configs/NYCTaxi.yaml -s 2
echo Experiment completed: Ks = 1
TIMEOUT 3


python main.py -g "pre_trained_random" -c "maskedAtt pre_trained_random, no sa" -cf configs/NYCTaxi.yaml
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "pre_trained_thresholded" -c "maskedAtt pre_trained_thresholded, no sa" -cf configs/NYCTaxi.yaml
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "pre_trained_symmetric" -c "maskedAtt pre_trained_symmetric, no sa" -cf configs/NYCTaxi.yaml
echo Experiment completed: Ks = 1
TIMEOUT 3




echo All experiments completed.
TIMEOUT 30000
