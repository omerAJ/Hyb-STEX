@echo off
setlocal enabledelayedexpansion

call "C:\ProgramData\Anaconda3\Scripts\activate.bat" sds-venv
TIMEOUT /T 3 /NOBREAK

set "EXP_COMMENT=nodewise soft-gpd tail correction"

call :run_dataset "configs/NYCTaxi.yaml" "NYCTaxi"
call :run_dataset "configs/NYCBike1.yaml" "NYCBike1"
call :run_dataset "configs/NYCBike2.yaml" "NYCBike2"
call :run_dataset "configs/BJTaxi.yaml" "BJTaxi"

echo All experiments completed.
TIMEOUT 99999

cmd /k
goto :eof

:run_dataset
set "CONFIG=%~1"
set "DATASET=%~2"
echo Starting experiments for !DATASET! using !CONFIG!

FOR /L %%G IN (1,1,5) DO (
    echo Running !DATASET! seed %%G
    python main.py -c "%EXP_COMMENT%" -s %%G -cf !CONFIG!
    echo Completed !DATASET! seed %%G
    TIMEOUT /T 3 /NOBREAK
)

goto :eof
