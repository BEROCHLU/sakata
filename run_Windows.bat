@echo off
@rem powershell -Command "Start-Process -NoNewWindow 'npm' -ArgumentList 'run main+batch'"
@rem powershell -Command "Start-Process -NoNewWindow 'npm' -ArgumentList 'run sakata+batch'"
start pwsh -NoExit -Command "node main-batch.js > ./result/nbatch.log; python ./python/plot-multi.py"
start pwsh -NoExit -Command "node braindevice-batch.js > ./result/sakata-batch.log"
