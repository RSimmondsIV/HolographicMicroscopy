# Create virtual environment if it doesn't exist
if (!(Test-Path ".venv")) {
    python -m venv .venv
}

# Activate it
& ".\.venv\Scripts\Activate.ps1"

# Upgrade pip
python -m pip install --upgrade pip

# Install required packages
pip install -r requirements.txt

Write-Host ""
Write-Host "Setup complete."
Write-Host "To run the project, use:"
Write-Host ".\.venv\Scripts\Activate.ps1"
Write-Host "python simplified_1d_laser_code.py"