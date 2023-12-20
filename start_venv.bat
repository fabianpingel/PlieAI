@echo off

REM ### Hier den Namen der virtuellen Umgebung angeben ###
set VENV=PlieAI_venv

echo Virtuelle Umgebung aktivieren
@echo on
call .\%VENV%\Scripts\activate
@echo off

REM ### Terminal öffnen
cmd

