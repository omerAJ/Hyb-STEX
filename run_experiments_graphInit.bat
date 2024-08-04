@echo off
setlocal enabledelayedexpansion

call activate ST-SSL
TIMEOUT 3



REM Loop through values 1 to 3 for the -s parameter
FOR /L %%G IN (1,1,3) DO (
    python main.py -c "bias wd -6, bias lr 10, cls lr times 0.01, z1 for bias, evs detached" -s %%G -cf configs/BJTaxi.yaml
    echo Experiment completed: Ks = %%G
    TIMEOUT /T 3 /NOBREAK
)



echo All experiments completed.
TIMEOUT 99999

cmd /k