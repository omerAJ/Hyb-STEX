## Environment

Use the Conda environment `sds-venv` for all repo work.

On this machine, plain `conda activate sds-venv` from PowerShell can fail because PowerShell script/module loading is blocked by execution policy. The working invocation is:

```powershell
cmd /d /c 'call C:\ProgramData\Anaconda3\condabin\conda.bat activate sds-venv && <command>'
```

Examples:

```powershell
cmd /d /c 'call C:\ProgramData\Anaconda3\condabin\conda.bat activate sds-venv && python -V'
cmd /d /c 'call C:\ProgramData\Anaconda3\condabin\conda.bat activate sds-venv && python main.py -cf configs/NYCTaxi.yaml'
```
