@echo off
setlocal enabledelayedexpansion

call activate ST-SSL
TIMEOUT 3



python main.py -c "using the pretrained classifier as initialization" -s 1 -lp "D:\omer\ST-SSL_simple2\experiments\NYCTaxi\pred__seed=1\20240720-123420\training only the classifier with ff proj and focal loss\best_model.pth"

python main.py -c "using the pretrained classifier as initialization" -s 2 -lp "D:\omer\ST-SSL_simple2\experiments\NYCTaxi\pred__seed=2\20240720-124547\training only the classifier with ff proj and focal loss\best_model.pth"

python main.py -c "using the pretrained classifier as initialization" -s 3 -lp "D:\omer\ST-SSL_simple2\experiments\NYCTaxi\pred__seed=3\20240720-125626\training only the classifier with ff proj and focal loss\best_model.pth"


echo All experiments completed.
TIMEOUT 99999

cmd /k