@echo off
set "PROJECT=F:\xueliu-mahjong-ai"
set "IMAGE_DIR=%PROJECT%\data\labeling\phone_landscape_round2_split_20260630\human_review\images_to_label"
set "CLASSES=%IMAGE_DIR%\classes.txt"

"%PROJECT%\.venv-labeling\Scripts\xanylabeling.exe" --no-auto-update-check --filename "%IMAGE_DIR%" --labels "%CLASSES%" --output "%IMAGE_DIR%" --autosave
