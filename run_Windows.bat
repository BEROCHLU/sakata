@echo off
powershell -Command "Start-Process -NoNewWindow 'npm' -ArgumentList 'run main+batch'"
powershell -Command "Start-Process -NoNewWindow 'npm' -ArgumentList 'run sakata+batch'"
powershell -Command "python app.py"
pause
