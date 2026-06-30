@echo off
set "PROJECT=F:\xueliu-mahjong-ai"
set "IMAGE_DIR=%PROJECT%\data\labeling\phone_landscape_20260629_154234_from_start\images_to_label"
set "CLASSES=%IMAGE_DIR%\classes.txt"

"%PROJECT%\.venv-labeling\Scripts\xanylabeling.exe" --no-auto-update-check --filename "%IMAGE_DIR%" --labels "%CLASSES%" --output "%IMAGE_DIR%" --autosave
