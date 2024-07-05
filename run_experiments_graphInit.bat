@echo off
setlocal enabledelayedexpansion

call activate ST-SSL
TIMEOUT 3




REM Loop through values 1 to 3 for the -s parameter
FOR /L %%G IN (1,1,5) DO (
    python main.py -g 8_neighbours -c "affinityx2 no sa" -s %%G -afc True -cf configs/NYCTaxi.yaml
    echo Experiment completed: Ks = %%G
    TIMEOUT /T 3 /NOBREAK
)

REM Loop through values 1 to 3 for the -s parameter
FOR /L %%G IN (1,1,5) DO (
    python main.py -g 8_neighbours -c "no affinityx2 no sa" -s %%G -cf configs/NYCTaxi.yaml
    echo Experiment completed: Ks = %%G
    TIMEOUT /T 3 /NOBREAK
)

REM Loop through values 1 to 3 for the -s parameter
FOR /L %%G IN (1,1,5) DO (
    python main.py -g 8_neighbours -c "affinityx2 no sa" -s %%G -afc True -cf configs/NYCBike1.yaml
    echo Experiment completed: Ks = %%G
    TIMEOUT /T 3 /NOBREAK
)

REM Loop through values 1 to 3 for the -s parameter
FOR /L %%G IN (1,1,5) DO (
    python main.py -g 8_neighbours -c "no affinityx2 no sa" -s %%G -cf configs/NYCBike1.yaml
    echo Experiment completed: Ks = %%G
    TIMEOUT /T 3 /NOBREAK
)

REM Loop through values 1 to 3 for the -s parameter
FOR /L %%G IN (1,1,5) DO (
    python main.py -g 8_neighbours -c "affinityx2 no sa" -s %%G -afc True -cf configs/NYCBike2.yaml
    echo Experiment completed: Ks = %%G
    TIMEOUT /T 3 /NOBREAK
)

REM Loop through values 1 to 3 for the -s parameter
FOR /L %%G IN (1,1,5) DO (
    python main.py -g 8_neighbours -c "no affinityx2 no sa" -s %%G -cf configs/NYCBike2.yaml
    echo Experiment completed: Ks = %%G
    TIMEOUT /T 3 /NOBREAK
)


echo All experiments completed.
TIMEOUT 99999

cmd /k