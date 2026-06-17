# PowerShell: Clean cache and run tests
# Usage: PowerShell -ExecutionPolicy Bypass -File fix_and_test.ps1

Write-Host "`n" -Foreground White
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "Cleaning Python cache..." -ForegroundColor Yellow
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "`n"

# Remove __pycache__ directories
Get-ChildItem -Path . -Filter "__pycache__" -Recurse -Directory |
    ForEach-Object {
        Write-Host "Removing: $_" -ForegroundColor Gray
        Remove-Item $_ -Recurse -Force -ErrorAction SilentlyContinue
    }

# Remove .pytest_cache directories
Get-ChildItem -Path . -Filter ".pytest_cache" -Recurse -Directory |
    ForEach-Object {
        Write-Host "Removing: $_" -ForegroundColor Gray
        Remove-Item $_ -Recurse -Force -ErrorAction SilentlyContinue
    }

Write-Host "`n"
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "Running pytest with fresh cache..." -ForegroundColor Yellow
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "`n"

# Run pytest
python -m pytest tests/ -v --tb=short

Write-Host "`n"
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "✅ Test run complete!" -ForegroundColor Green
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "`n"
