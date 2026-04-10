@echo off
setlocal enabledelayedexpansion

call "C:\ProgramData\Anaconda3\Scripts\activate.bat" sds-venv
TIMEOUT 3

@REM REM Loop through values 1 to 5 for the -s parameter
@REM FOR /L %%G IN (1,1,3) DO (
@REM     python main.py -c "fix phase-wise training" -s %%G -cf configs/NYCBike1.yaml 
@REM     echo Experiment completed: Ks = %%G
@REM     TIMEOUT /T 3 /NOBREAK
@REM )


REM Loop through values 1 to 5 for the -s parameter
FOR /L %%G IN (1,1,5) DO (
    python main.py -c "fix phase-wise training evs_90" -s %%G -cf configs/NYCTaxi.yaml 
    echo Experiment completed: Ks = %%G
    TIMEOUT /T 3 /NOBREAK
)



@REM REM Loop through values 1 to 5 for the -s parameter
@REM FOR /L %%G IN (1,1,3) DO (
@REM     python main.py -c "fix phase-wise training" -s %%G -cf configs/NYCBike2.yaml 
@REM     echo Experiment completed: Ks = %%G
@REM     TIMEOUT /T 3 /NOBREAK
@REM )

echo All experiments completed.
TIMEOUT 99999

cmd /k
