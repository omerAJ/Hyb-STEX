@echo off
setlocal enabledelayedexpansion

call activate i-jepaVENV
TIMEOUT 3



python main.py -g "no_sconv" -c "no_sconv, combined layer 128, then 2xSA with skip, layer norms after sa, 2 heads" -cf configs/NYCTaxi.yaml
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "zeros" -c "zero init graph conv, combined layer 128, then 2xSA with skip, layer norms after sa, 2 heads" -cf configs/NYCTaxi.yaml
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "8_neighbours" -c "8_neighbours init graph conv, combined layer 128, then 2xSA with skip, layer norms after sa, 2 heads" -cf configs/NYCTaxi.yaml
echo Experiment completed: Ks = 1
TIMEOUT 3


python main.py -g "no_sconv" -c "no_sconv, ff on combined layer, then 2xSA with skip, layer norms after ff and sa, 2 heads and d_ff=2" -cf configs/NYCTaxi.yaml -ff True
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "zeros" -c "zero init graph conv, ff on combined layer, then 2xSA with skip, layer norms after ff and sa, 2 heads and d_ff=2" -cf configs/NYCTaxi.yaml -ff True
echo Experiment completed: Ks = 1
TIMEOUT 3

python main.py -g "8_neighbours" -c "8_neighbours init graph conv, then 2xSA with skip, layer norms after ff and sa, 2 heads and d_ff=2" -cf configs/NYCTaxi.yaml -ff True
echo Experiment completed: Ks = 1
TIMEOUT 3


echo All experiments completed.
TIMEOUT 3
