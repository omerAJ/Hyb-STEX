@echo off
setlocal enabledelayedexpansion

call "C:\Users\IST\miniconda3\Scripts\activate.bat"
call conda activate ST-SSL
TIMEOUT 3



FOR %%G IN (4 5) DO (
    python main.py -c "evl (cheating) 90 percentile evs" -s %%G -cf configs/NYCBike2.yaml -l "mae"
    echo Experiment completed: Ks = %%G
    TIMEOUT /T 3 /NOBREAK
)

FOR %%G IN (4 5) DO (
    python main.py -c "evl (cheating) 90 percentile evs" -s %%G -cf configs/NYCBike1.yaml -l "mae"
    echo Experiment completed: Ks = %%G
    TIMEOUT /T 3 /NOBREAK
)


echo All experiments completed.
TIMEOUT 99999

cmd /k