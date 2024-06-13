@echo off
@rem powershell -Command "Start-Process -NoNewWindow 'npm' -ArgumentList 'run main+batch'"
@rem powershell -Command "Start-Process -NoNewWindow 'npm' -ArgumentList 'run sakata+batch'"
@rem start pwsh -NoExit -Command "node ./nodejs/main-batch.js > ./result/output1.log"
@rem start pwsh -NoExit -Command "node ./nodejs/braindevice-batch.js > ./result/output2.log; python ./util/plot-multi.py"
start pwsh -Command "node ./nodejs/main-batch.js | Out-File -FilePath ./result/output1.log"
start pwsh -NoExit -Command "node ./nodejs/braindevice-batch.js | Out-File -FilePath ./result/output2.log; python ./util/plot-multi.py"