@echo off
setlocal enabledelayedexpansion

REM ============================================
REM YAML Batch Runner Script
REM ============================================
REM Usage: run_yaml_batch.bat [sleep_seconds]
REM Example: run_yaml_batch.bat 30
REM ============================================

REM Set default sleep time (in seconds)
set SLEEP_TIME=30
if not "%~1"=="" set SLEEP_TIME=%~1

echo ============================================
echo YAML Batch Runner
echo ============================================
echo Sleep time between runs: %SLEEP_TIME% seconds
echo.

REM ============================================
REM CONFIGURE YOUR YAML FILES HERE
REM ============================================
REM Add your YAML file paths below (one per line)
REM Make sure to use the correct path format for Windows

set YAML_FILES[0]=ditec_wdn_dataset\arguments\Net3_sim_configactor_muleval22_head_pump+base_speed_0.92_c10_e1.yaml
set YAML_FILES[1]=ditec_wdn_dataset\arguments\Net3_sim_configactor_muleval22_head_pump+base_speed_0.70_c3_e0.yaml
set YAML_FILES[2]=ditec_wdn_dataset\arguments\Net3_sim_configactor_muleval22_head_pump+initial_status_0.96_c15_e1.yaml
set YAML_FILES[3]=ditec_wdn_dataset\arguments\Net3_sim_configactor_muleval22_head_pump+initial_status_0.99_c6_e0.yaml
set YAML_FILES[4]=ditec_wdn_dataset\arguments\Net3_sim_configactor_muleval22_head_pump+pump_curve_name_y_0.30_c0_e0.yaml
set YAML_FILES[5]=ditec_wdn_dataset\arguments\Net3_sim_configactor_muleval22_head_pump+pump_curve_name_y_0.90_c8_e1.yaml
set YAML_FILES[6]=ditec_wdn_dataset\arguments\Net3_sim_configactor_muleval22_junction+base_demand_0.35_c2_e0.yaml
set YAML_FILES[7]=ditec_wdn_dataset\arguments\Net3_sim_configactor_muleval22_junction+base_demand_0.91_c12_e1.yaml
set YAML_FILES[8]=ditec_wdn_dataset\arguments\Net3_sim_configactor_muleval22_pipe+roughness_0.93_c7_e0.yaml
set YAML_FILES[9]=ditec_wdn_dataset\arguments\Net3_sim_configactor_muleval22_pipe+roughness_0.93_c13_e1.yaml
set YAML_FILES[10]=ditec_wdn_dataset\arguments\Net3_sim_configactor_muleval22_reservoir+base_head_0.96_c5_e0.yaml
set YAML_FILES[11]=ditec_wdn_dataset\arguments\Net3_sim_configactor_muleval22_reservoir+base_head_0.96_c9_e1.yaml
set YAML_FILES[12]=ditec_wdn_dataset\arguments\Net3_sim_configactor_muleval22_tank+init_level_0.96_c4_e0.yaml
set YAML_FILES[13]=ditec_wdn_dataset\arguments\Net3_sim_configactor_muleval22_tank+init_level_0.97_c14_e1.yaml
set TOTAL_FILES=14

REM ============================================
REM OPTIONAL: Configure simulation parameters
REM ============================================
REM You can modify these parameters as needed
set TASK=sim
set NUM_SAMPLES=100
set VERBOSE=true

echo Total files to process: %TOTAL_FILES%
echo Task: %TASK%
echo Number of samples: %NUM_SAMPLES%
echo Verbose: %VERBOSE%
echo.

REM ============================================
REM RUN SIMULATIONS SEQUENTIALLY
REM ============================================
echo Starting batch simulation...
echo.

set SUCCESS_COUNT=0
set FAILED_COUNT=0

set /a VALIDATE_END=%TOTAL_FILES%-1
for /L %%i in (0,1,%VALIDATE_END%) do (
    set CURRENT_FILE=!YAML_FILES[%%i]!
    set /a FILE_NUM=%%i+1
    
    echo ============================================
    echo Processing file %FILE_NUM%/%TOTAL_FILES%
    echo File: !CURRENT_FILE!
    echo Time: %date% %time%
    echo ============================================
    
    REM Run the simulation
    python main.py --task %TASK% --yaml_path "!CURRENT_FILE!"
    
    REM Check if the command was successful
    if !errorlevel! equ 0 (
        echo SUCCESS: File %FILE_NUM% completed successfully
        set /a SUCCESS_COUNT+=1
    ) else (
        echo ERROR: File %FILE_NUM% failed with error code !errorlevel!
        set /a FAILED_COUNT+=1
    )
    
    echo.
    
    REM Sleep between runs (except for the last one)
    if %%i lss %VALIDATE_END% (
        echo Waiting %SLEEP_TIME% seconds before next run...
        timeout /t %SLEEP_TIME% /nobreak >nul
        echo.
    )
)

REM ============================================
REM FINAL SUMMARY
REM ============================================
echo ============================================
echo BATCH PROCESSING COMPLETE
echo ============================================
echo Total files processed: %TOTAL_FILES%
echo Successful: %SUCCESS_COUNT%
echo Failed: %FAILED_COUNT%
echo.

if %FAILED_COUNT% gtr 0 (
    echo WARNING: Some files failed to process.
    echo Check the error messages above for details.
) else (
    echo All files processed successfully!
)

echo.
echo Batch processing finished at: %date% %time%
pause 