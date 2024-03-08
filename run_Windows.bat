@echo off
@rem powershell -Command "Start-Process -NoNewWindow 'npm' -ArgumentList 'run main+batch'"
@rem powershell -Command "Start-Process -NoNewWindow 'npm' -ArgumentList 'run sakata+batch'"
start pwsh -NoExit -Command "node ./nodejs/main-batch.js > ./result/main-batch.log"
start pwsh -NoExit -Command "node ./nodejs/braindevice-batch.js > ./result/sakata-batch.log; python ./util/plot-multi.py"
