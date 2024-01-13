Set VENV=MA_venv

call .\%VENV%\Scripts\activate.bat

cd C:\Users\Fabian Pingel\Downloads\ssl-proxy-windows-amd64.exe

ssl-proxy-windows-amd64.exe -from 0.0.0.0:8000 -to 127.0.0.1:8501

pause