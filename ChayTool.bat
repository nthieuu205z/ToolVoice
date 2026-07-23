@echo off
rem ============================================================
rem  ToolVietSub - chay bang 1 cu nhap chuot
rem  Nhap doi (double-click) file nay de bat cong cu.
rem ============================================================
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
  echo.
  echo [LOI] Khong tim thay moi truong .venv trong thu muc nay.
  echo       Hay cai dat theo HUONG_DAN_WINDOWS.md muc 3 truoc khi chay.
  echo.
  pause
  exit /b 1
)

call ".venv\Scripts\activate.bat"

rem Mo trinh duyet sau 5 giay (cho server kip khoi dong)
start "mo-trinh-duyet" /min cmd /c "timeout /t 5 >nul & start http://localhost:8000"

echo.
echo ============================================================
echo   ToolVietSub dang khoi dong...
echo   Giao dien: http://localhost:8000  (tu mo sau vai giay)
echo   Tat server: dong cua so nay hoac bam Ctrl+C.
echo ============================================================
echo.

python -m uvicorn backend.main:app --port 8000

echo.
echo Server da dung. Nhan phim bat ky de dong cua so.
pause >nul
