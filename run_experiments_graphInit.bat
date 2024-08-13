@echo off
setlocal enabledelayedexpansion

call "C:\Users\IST\miniconda3\Scripts\activate.bat"
call conda activate ST-SSL
TIMEOUT 3



REM Loop through values 1 to 3 for the -s parameter
FOR /L %%G IN (1,1,5) DO (
    python main.py -c "no_sconv train" -s %%G -cf configs/BJTaxi.yaml -g "no_sconv"
    echo Experiment completed: Ks = %%G
    TIMEOUT /T 3 /NOBREAK
)



echo All experiments completed.
TIMEOUT 99999

cmd /k