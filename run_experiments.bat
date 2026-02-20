@echo off
setlocal enabledelayedexpansion

CALL C:\Users\PCF\anaconda3\Scripts\activate.bat
call conda activate sds-venv
TIMEOUT 3


REM Loop through values 1 to 5 for the -s parameter
FOR /L %%G IN (1,1,3) DO (
    python main.py -c "GPD-Experiment3-useTrainThreshold&FilterBeforeThreshold" -s %%G -cf configs/NYCTaxi.yaml 
    echo Experiment completed: Ks = %%G
    TIMEOUT /T 3 /NOBREAK
)


REM Loop through values 1 to 5 for the -s parameter
FOR /L %%G IN (1,1,3) DO (
    python main.py -c "GPD-Experiment3-useTrainThreshold&FilterBeforeThreshold" -s %%G -cf configs/NYCBike1.yaml 
    echo Experiment completed: Ks = %%G
    TIMEOUT /T 3 /NOBREAK
)


REM Loop through values 1 to 5 for the -s parameter
FOR /L %%G IN (1,1,3) DO (
    python main.py -c "GPD-Experiment3-useTrainThreshold&FilterBeforeThreshold" -s %%G -cf configs/NYCBike2.yaml 
    echo Experiment completed: Ks = %%G
    TIMEOUT /T 3 /NOBREAK
)


REM Loop through values 1 to 5 for the -s parameter
FOR /L %%G IN (1,1,3) DO (
    python main.py -c "GPD-Experiment3-useTrainThreshold&FilterBeforeThreshold" -s %%G -cf configs/BJTaxi.yaml 
    echo Experiment completed: Ks = %%G
    TIMEOUT /T 3 /NOBREAK
)

echo All experiments completed.
TIMEOUT 99999

cmd /k