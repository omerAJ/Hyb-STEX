@echo off
setlocal enabledelayedexpansion

call activate ST-SSL
TIMEOUT 3



python main.py -c "using the pretrained classifier as initialization" -s 1 -lp "D:\omer\ST-SSL\experiments\NYCTaxi\pred__seed=1\20240720-123431\training only the classifier with linear proj and focal loss\best_model.pth"

python main.py -c "using the pretrained classifier as initialization" -s 2 -lp "D:\omer\ST-SSL\experiments\NYCTaxi\pred__seed=2\20240720-124507\training only the classifier with linear proj and focal loss\best_model.pth"

python main.py -c "using the pretrained classifier as initialization" -s 3 -lp "D:\omer\ST-SSL\experiments\NYCTaxi\pred__seed=3\20240720-125741\training only the classifier with linear proj and focal loss\best_model.pth"


echo All experiments completed.
TIMEOUT 99999

cmd /k