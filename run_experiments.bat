@echo off
setlocal enabledelayedexpansion

call activate ST-SSL
TIMEOUT 3


python main.py -c "just finetune bias" -s 1 -cf configs/NYCTaxi.yaml -lp "D:\omer\ST-SSL\experiments\NYCTaxi\pred__seed=1\20240805-231218\cls lr 0.1, wd bias minus 5, annealing\best_model.pth"
    
python main.py -c "just finetune bias" -s 2 -cf configs/NYCTaxi.yaml -lp "D:\omer\ST-SSL\experiments\NYCTaxi\pred__seed=2\20240806-001022\cls lr 0.1, wd bias minus 5, annealing\best_model.pth"

python main.py -c "just finetune bias" -s 3 -cf configs/NYCTaxi.yaml -lp "D:\omer\ST-SSL\experiments\NYCTaxi\pred__seed=3\20240806-005523\cls lr 0.1, wd bias minus 5, annealing\best_model.pth"


python main.py -c "just finetune bias" -s 1 -cf configs/NYCBike1.yaml -lp "D:\omer\ST-SSL\experiments\NYCBike1\pred__seed=1\20240805-231242\cls lr 0.1, wd bias minus 5, annealing\best_model.pth"

python main.py -c "just finetune bias" -s 2 -cf configs/NYCBike1.yaml -lp "D:\omer\ST-SSL\experiments\NYCBike1\pred__seed=2\20240806-010058\cls lr 0.1, wd bias minus 5, annealing\best_model.pth"

python main.py -c "just finetune bias" -s 3 -cf configs/NYCBike1.yaml -lp "D:\omer\ST-SSL\experiments\NYCBike1\pred__seed=3\20240806-024255\cls lr 0.1, wd bias minus 5, annealing\best_model.pth"


python main.py -c "just finetune bias" -s 1 -cf configs/NYCBike2.yaml -lp "D:\omer\ST-SSL\experiments\NYCBike2\pred__seed=1\20240806-015146\cls lr 0.1, wd bias minus 5, annealing\best_model.pth"

python main.py -c "just finetune bias" -s 2 -cf configs/NYCBike2.yaml -lp "D:\omer\ST-SSL\experiments\NYCBike2\pred__seed=2\20240806-110112\cls lr 0.1, wd bias minus 5, annealing, XTY plus B\best_model.pth"

python main.py -c "just finetune bias" -s 3 -cf configs/NYCBike2.yaml -lp "D:\omer\ST-SSL\experiments\NYCBike2\pred__seed=3\20240806-030309\cls lr 0.1, wd bias minus 5, annealing\best_model.pth"

echo All experiments completed.
TIMEOUT 99999

cmd /k