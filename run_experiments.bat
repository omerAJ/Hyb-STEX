@echo off
setlocal enabledelayedexpansion

REM Define different S_Loss and T_Loss values
set random_seeds= 1 2 3
set S_Losses=0 1
set T_Losses=0 1

call activate i-jepaVENV
TIMEOUT 3


REM Loop through each combination of S_Loss and T_Loss
for %%r in (%random_seeds%) do (
    for %%s in (%S_Losses%) do (
        for %%t in (%T_Losses%) do (
            echo Running experiment with S_Loss=%%s and T_Loss=%%t and random seed=%%r
            python main.py --S_Loss %%s --T_Loss %%t --seed %%r
            echo Experiment completed: S_Loss=%%s, T_Loss=%%t
            TIMEOUT 3
        )
        
    )
)
echo All experiments completed.
TIMEOUT 3
