@echo off
rem Run this from anywhere once it's set up on your PATH (see below), or
rem just double-click it. It always starts server.py from the folder this
rem file sits in, regardless of where cmd's working directory started.
cd /d "%~dp0"
python server.py
