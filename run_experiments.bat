@echo off
setlocal enabledelayedexpansion

call activate ST-SSL
TIMEOUT 3


python main.py -c "pwt 4ps t2" -s 1 -cf configs/NYCBike2.yaml

echo exp 1 complete


python main.py -c "pwt 4ps t2" -s 1 -cf configs/NYCBike1.yaml
    
echo All experiments completed.
TIMEOUT 99999

cmd /k