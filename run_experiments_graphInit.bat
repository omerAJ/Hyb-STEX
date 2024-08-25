@echo off
setlocal enabledelayedexpansion

call "C:\Users\IST\miniconda3\Scripts\activate.bat"
call conda activate ST-SSL
TIMEOUT 3


FOR %%G IN (1 2 3) DO (
    python main.py -c "common backbone ablation bias learnable global but not input dependent" -s %%G -cf configs/NYCBike2.yaml -l "mae"
    echo Experiment completed: Ks = %%G
    TIMEOUT /T 3 /NOBREAK
)

FOR /L %%G IN (1,1,5) DO (
    python main.py -c "common backbone ablation bias learnable global but not input dependent" -s %%G -cf configs/NYCTaxi.yaml -l "mae"
    echo Experiment completed: Ks = %%G
    TIMEOUT /T 3 /NOBREAK
)




echo All experiments completed.
TIMEOUT 99999

cmd /k