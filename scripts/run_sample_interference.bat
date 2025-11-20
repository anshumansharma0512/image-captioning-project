@echo off
echo Running image captioning inference...

if "%1"=="" (
    echo Usage: run_sample_inference.bat image_path checkpoint_path
    exit /b
)

python evaluate.py --img_path "%1" --checkpoint "%2"
