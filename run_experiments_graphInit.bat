@echo off
setlocal enabledelayedexpansion

call activate ST-SSL
TIMEOUT 3



REM Loop through values 1 to 3 for the -s parameter
FOR /L %%G IN (1,1,3) DO (
    python main.py -c "train cls after 15, bias wd -5, bias lr times 0.1, independent learnable vector for each node" -s %%G -cf configs/NYCBike1.yaml
    echo Experiment completed: Ks = %%G
    TIMEOUT /T 3 /NOBREAK
)



echo All experiments completed.
TIMEOUT 99999

cmd /k