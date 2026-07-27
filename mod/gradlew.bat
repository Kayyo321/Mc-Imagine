@echo off
setlocal
echo Downloading and setting up Gradle...
set GRADLE_CMD=gradle
%GRADLE_CMD% %*
