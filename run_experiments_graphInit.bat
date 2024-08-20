@echo off
setlocal enabledelayedexpansion

call "C:\Users\IST\miniconda3\Scripts\activate.bat"
call conda activate ST-SSL
TIMEOUT 3


FOR /L %%G IN (1,1,3) DO (
    python main.py -c "frechet sconv with skip (learnable weights), train v3" -s %%G -cf configs/NYCTaxi.yaml -l "frechet"
    echo Experiment completed: Ks = %%G
    TIMEOUT /T 3 /NOBREAK
)




echo All experiments completed.
TIMEOUT 99999

cmd /k