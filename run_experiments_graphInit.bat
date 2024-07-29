@echo off
setlocal enabledelayedexpansion

call activate ST-SSL
TIMEOUT 3



REM Loop through values 1 to 3 for the -s parameter
FOR /L %%G IN (1,1,3) DO (
    python main.py -c "evl classifier, 90perc, using pred as feature to classifier with sep backbone" -s %%G -cf configs/NYCTaxi.yaml
    echo Experiment completed: Ks = %%G
    TIMEOUT /T 3 /NOBREAK
)



echo All experiments completed.
TIMEOUT 99999

cmd /k