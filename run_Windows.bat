@echo off
@rem powershell -Command "Start-Process -NoNewWindow 'npm' -ArgumentList 'run main+batch'"
@rem powershell -Command "Start-Process -NoNewWindow 'npm' -ArgumentList 'run sakata+batch'"
start pwsh -Command "npm run main+batch; python ./python/plot-multi.py"
start pwsh -Command "npm run sakata+batch"
