@echo off
IF NOT EXIST "stb" (
    echo Cloning stb repository...
    git clone https://github.com/nothings/stb.git
) ELSE (
    echo stb folder already exists, skipping clone.
)
pause
