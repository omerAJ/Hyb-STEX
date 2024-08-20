@echo off
setlocal enabledelayedexpansion

call "C:\Users\IST\miniconda3\Scripts\activate.bat"
call conda activate ST-SSL
TIMEOUT 3


FOR /L %%G IN (1,1,5) DO (
    python main.py -c "gumbell sconv with skip (learnable weights), train v4" -s %%G -cf configs/NYCTaxi.yaml -l "gumbell"
    echo Experiment completed: Ks = %%G
    TIMEOUT /T 3 /NOBREAK
)




FOR /L %%G IN (1,1,5) DO (
    python main.py -c "gumbell sconv with skip (learnable weights), train v4" -s %%G -cf configs/NYCBike1.yaml -l "gumbell"
    echo Experiment completed: Ks = %%G
    TIMEOUT /T 3 /NOBREAK
)



FOR /L %%G IN (1,1,5) DO (
    python main.py -c "gumbell sconv with skip (learnable weights), train v3" -s %%G -cf configs/NYCBike2.yaml -l "gumbell"
    echo Experiment completed: Ks = %%G
    TIMEOUT /T 3 /NOBREAK
)




FOR /L %%G IN (1,1,5) DO (
    python main.py -c "gumbell sconv with skip (learnable weights), train v3" -s %%G -cf configs/BJTaxi.yaml -l "gumbell"
    echo Experiment completed: Ks = %%G
    TIMEOUT /T 3 /NOBREAK
)


echo All experiments completed.
TIMEOUT 99999

cmd /k