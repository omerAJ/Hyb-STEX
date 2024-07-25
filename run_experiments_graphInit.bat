@echo off
setlocal enabledelayedexpansion

call activate ST-SSL
TIMEOUT 3



REM Loop through values 1 to 3 for the -s parameter
FOR /L %%G IN (1,1,3) DO (
    python main.py -c "pretrained classifier @0.35, mlp_bias" -s %%G -cf configs/NYCTaxi.yaml -lp "D:\omer\ST-SSL_simple2\pretrain_experiments\NYCTaxi\pred__seed=1\20240724-154309\noComment\best_model.pth"
    echo Experiment completed: Ks = %%G
    TIMEOUT /T 3 /NOBREAK
)



echo All experiments completed.
TIMEOUT 99999

cmd /k