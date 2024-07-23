@echo off
setlocal enabledelayedexpansion

call activate i-jepaVENV
TIMEOUT 3



REM Loop through values 1 to 3 for the -s parameter
FOR /L %%G IN (1,1,3) DO (
    python main.py -c "evl classifier, uniform init learnable vectors" -s %%G -cf configs/NYCBike1.yaml
    echo Experiment completed: Ks = %%G
    TIMEOUT /T 3 /NOBREAK
)


REM Loop through values 1 to 3 for the -s parameter
FOR /L %%G IN (1,1,3) DO (
    python main.py -c "evl classifier, uniform init learnable vectors" -s %%G -cf configs/NYCBike2.yaml
    echo Experiment completed: Ks = %%G
    TIMEOUT /T 3 /NOBREAK
)


REM Loop through values 1 to 3 for the -s parameter
FOR /L %%G IN (1,1,3) DO (
    python main.py -c "evl classifier, uniform init learnable vectors" -s %%G -cf configs/BJTaxi.yaml
    echo Experiment completed: Ks = %%G
    TIMEOUT /T 3 /NOBREAK
)

echo All experiments completed.
TIMEOUT 99999

cmd /k