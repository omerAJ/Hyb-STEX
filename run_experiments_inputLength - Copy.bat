@echo off
setlocal enabledelayedexpansion

set random_seeds= 1 2 3
set input_length= A B C D
set input_seq= 35
call activate i-jepaVENV
TIMEOUT 3


REM Loop through each combination of S_Loss and T_Loss
for %%r in (%random_seeds%) do (
    for %%l in (%input_length%) do (
        echo Running experiment with random seed=%%r and input_sequence_type=%%l and input_dataset_context=%%input_seq	
	TIMEOUT 3        
	python main_NB2.py --seed %%r --input_sequence_type %%l --input_dataset_context %input_seq%  
        echo Experiment completed: seed=%%r input_length=%%l
        TIMEOUT 3
    )
)
echo All experiments completed.
TIMEOUT 3
