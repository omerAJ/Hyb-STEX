@echo off
setlocal enabledelayedexpansion

set random_seeds= 1 2 3 4 5
set input_length= 19 14 9 4
call activate i-jepaVENV
TIMEOUT 3


REM Loop through each combination of S_Loss and T_Loss
for %%r in (%random_seeds%) do (
    for %%l in (%input_length%) do (
        
        echo Running experiment with random seed=%%r and input_length=%%l 
	TIMEOUT 3        
	python main.py --seed %%r --input_length %%l
        echo Experiment completed: seed=%%r input_length=%%l
        TIMEOUT 3
        
        
    )
)
echo All experiments completed.
TIMEOUT 3
