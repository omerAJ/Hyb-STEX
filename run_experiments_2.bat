@echo off
setlocal enabledelayedexpansion

call "C:\Users\IST\miniconda3\Scripts\activate.bat"
call conda activate ST-SSL
TIMEOUT 3


REM Loop through values 1 to 5 for the -s parameter
FOR /L %%G IN (4,1,5) DO (
    python main.py -c "training classifier but only to improve representation" -s %%G -cf configs/NYCTaxi.yaml -v "cls" 
    echo Experiment completed: Ks = %%G
    TIMEOUT /T 3 /NOBREAK
)


REM Loop through values 1 to 5 for the -s parameter
FOR /L %%G IN (1,1,5) DO (
    python main.py -c "adding bias" -s %%G -cf configs/NYCTaxi.yaml -v "bias" 
    echo Experiment completed: Ks = %%G
    TIMEOUT /T 3 /NOBREAK
)

echo All experiments completed.
TIMEOUT 99999

cmd /k

