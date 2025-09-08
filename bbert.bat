@echo off
REM BBERT - Windows batch executable for BERT-based DNA sequence analysis
REM Usage: bbert.bat file1.fasta file2.fastq.gz --output_dir results [options]

setlocal enabledelayedexpansion

REM Function to print colored output (Windows 10+)
set "RED=[91m"
set "GREEN=[92m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "NC=[0m"

REM Print functions
:print_error
echo %RED%❌ Error: %~1%NC% >&2
goto :eof

:print_success
echo %GREEN%✅ %~1%NC%
goto :eof

:print_warning
echo %YELLOW%⚠️  Warning: %~1%NC%
goto :eof

:print_info
echo %BLUE%ℹ️  %~1%NC%
goto :eof

REM Check if command exists
:command_exists
where %1 >nul 2>&1
goto :eof

REM Check Python environment
:check_python_env
call :print_info "Checking Python environment..."

REM Check if python is available
call :command_exists python
if errorlevel 1 (
    call :print_error "Python is not installed or not in PATH"
    call :print_info "Please install Python 3.10+ and try again"
    exit /b 1
)

REM Check Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set python_version=%%i
for /f "tokens=1,2 delims=." %%a in ("!python_version!") do (
    set major_version=%%a
    set minor_version=%%b
)

if !major_version! LSS 3 (
    call :print_error "Python !python_version! detected. BBERT requires Python 3.10+"
    exit /b 1
)
if !major_version! EQU 3 if !minor_version! LSS 10 (
    call :print_error "Python !python_version! detected. BBERT requires Python 3.10+"
    exit /b 1
)

call :print_success "Python !python_version! detected"
exit /b 0

REM Check conda environment
:check_conda_env
if defined CONDA_DEFAULT_ENV (
    if not "!CONDA_DEFAULT_ENV!"=="base" (
        call :print_success "Conda environment: !CONDA_DEFAULT_ENV!"
    ) else (
        call :print_warning "You're in the base conda environment"
        call :print_info "Consider activating a BBERT environment: conda activate BBERT"
    )
) else (
    call :command_exists conda
    if not errorlevel 1 (
        call :print_info "Conda available but no environment activated"
    )
)
exit /b 0

REM Check required Python packages
:check_python_packages
call :print_info "Checking required Python packages..."

set "required_packages=torch transformers pandas numpy pyarrow scikit-learn biopython"
set "missing_packages="

for %%p in (%required_packages%) do (
    python -c "import %%p" 2>nul
    if errorlevel 1 (
        set "missing_packages=!missing_packages! %%p"
    )
)

if not "!missing_packages!"=="" (
    call :print_error "Missing required packages:!missing_packages!"
    call :print_info "Please install missing packages or activate the BBERT environment"
    call :print_info "To install: pip install!missing_packages!"
    exit /b 1
)

call :print_success "All required packages found"
exit /b 0

REM Check model files
:check_model_files
call :print_info "Checking BBERT model files..."

set "model_dirs=models\diverse_bact_12_768_6_20000 emb_class_bact\models\emb_class_model_768H_3906K_80e emb_class_frame\models emb_class_coding\models\emb_coding_model_768_3906K_50e"
set "missing_models="

for %%d in (%model_dirs%) do (
    if not exist "%%d" (
        set "missing_models=!missing_models! %%d"
    )
)

if not "!missing_models!"=="" (
    call :print_error "Missing model directories:"
    for %%d in (!missing_models!) do echo   - %%d
    call :print_info "Please ensure you've downloaded the models using Git LFS:"
    call :print_info "  git lfs install"
    call :print_info "  git lfs pull"
    exit /b 1
)

call :print_success "Model files found"
exit /b 0

REM Check GPU availability
:check_gpu
call :print_info "Checking GPU availability..."

REM Check for NVIDIA GPU
call :command_exists nvidia-smi
if not errorlevel 1 (
    nvidia-smi >nul 2>&1
    if not errorlevel 1 (
        for /f "skip=1 tokens=*" %%i in ('nvidia-smi --query-gpu^=name --format^=csv^,noheader^,nounits') do (
            call :print_success "NVIDIA GPU detected: %%i"
            exit /b 0
        )
    )
)

call :print_warning "No GPU acceleration detected - will use CPU"
call :print_info "This will be slower but still functional"
exit /b 0

REM Validate input files
:validate_input_files
set "file_count=0"
set "invalid_files="

REM Count and validate files
for %%f in (%*) do (
    if not exist "%%f" (
        set "invalid_files=!invalid_files! %%f"
    ) else (
        set /a file_count+=1
    )
)

if !file_count! EQU 0 (
    call :print_error "No valid input files found"
    exit /b 1
)

if not "!invalid_files!"=="" (
    call :print_error "Input files not found:!invalid_files!"
    exit /b 1
)

call :print_success "!file_count! input file(s) found"
exit /b 0

REM Show usage
:show_usage
echo BBERT - BERT for Bacterial DNA Classification
echo.
echo USAGE:
echo     bbert.bat ^<input_files...^> --output_dir ^<directory^> [OPTIONS]
echo.
echo EXAMPLES:
echo     REM Single file
echo     bbert.bat example\sample.fasta --output_dir results
echo.
echo     REM Multiple files
echo     bbert.bat file1.fasta file2.fastq.gz --output_dir results --batch_size 512
echo.
echo     REM With embeddings (large output files)
echo     bbert.bat example\*.fasta.gz --output_dir results --emb_out
echo.
echo OPTIONS:
echo     --output_dir DIR    Directory to save output files (required)
echo     --batch_size N      Batch size for processing (default: 1024)
echo     --emb_out          Include sequence embeddings in output (warning: large files)
echo     --help             Show this help message
echo     --check            Run system checks only (don't process files)
echo.
echo SYSTEM REQUIREMENTS:
echo     - Python 3.10+
echo     - PyTorch, Transformers, BioPython, pandas, numpy, pyarrow
echo     - Git LFS for model files
echo     - GPU recommended but not required
echo.
echo For more information, see: https://github.com/AmirErez/BBERT
goto :eof

REM Main execution
:main
REM Check for help or check flags
for %%a in (%*) do (
    if "%%a"=="--help" goto show_usage
    if "%%a"=="-h" goto show_usage
    if "%%a"=="--check" goto run_checks
)

REM Show header
echo.
echo 🧬 BBERT - BERT for Bacterial DNA Classification
echo ==================================================
echo.

REM Run system checks
call :check_python_env
if errorlevel 1 exit /b 1

call :check_conda_env
call :check_python_packages
if errorlevel 1 exit /b 1

call :check_model_files
if errorlevel 1 exit /b 1

call :check_gpu
echo.

REM Parse arguments
set "input_files="
set "other_args="
set "has_output_dir="

:parse_args
if "%1"=="" goto done_parsing
if "%1"=="--output_dir" (
    set "other_args=!other_args! %1 %2"
    set "has_output_dir=1"
    shift
    shift
    goto parse_args
)
if "%1"=="--batch_size" (
    set "other_args=!other_args! %1 %2"
    shift
    shift
    goto parse_args
)
if "%1"=="--emb_out" (
    set "other_args=!other_args! %1"
    shift
    goto parse_args
)
if "%1" NEQ "" (
    if "%1:~0,2%"=="--" (
        call :print_error "Unknown option: %1"
        echo.
        goto show_usage
    ) else (
        set "input_files=!input_files! %1"
        shift
        goto parse_args
    )
)
shift
goto parse_args

:done_parsing
REM Validate input files
call :validate_input_files %input_files%
if errorlevel 1 exit /b 1
echo.

REM Check if output_dir is specified
if not defined has_output_dir (
    call :print_error "Missing required argument: --output_dir"
    echo.
    goto show_usage
)

REM Run BBERT inference
call :print_info "Starting BBERT inference..."
echo.

python source\inference.py %input_files% %other_args%

if errorlevel 1 (
    echo.
    call :print_error "BBERT analysis failed"
    exit /b 1
) else (
    echo.
    call :print_success "BBERT analysis completed successfully!"
)

exit /b 0

:run_checks
call :print_info "Running system checks..."
echo.
call :check_python_env
if errorlevel 1 exit /b 1
call :check_conda_env
call :check_python_packages
if errorlevel 1 exit /b 1
call :check_model_files
if errorlevel 1 exit /b 1
call :check_gpu
echo.
call :print_success "All system checks passed! BBERT is ready to use."
exit /b 0

REM Check if we're in the right directory
if not exist "source\inference.py" (
    call :print_error "BBERT inference script not found"
    call :print_info "Please run this script from the BBERT root directory"
    exit /b 1
)

REM Run main function
goto main