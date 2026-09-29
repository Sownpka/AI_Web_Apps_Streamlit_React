@echo off
REM Chạy backend FastAPI (kèm React build) ở cổng 8000 và Streamlit ở cổng 8501.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set YOLO_AUTOINSTALL=false
start "AI Web Apps - API" cmd /k python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
start "AI Web Apps - Streamlit" cmd /k "set API_URL=http://localhost:8000&& python -m streamlit run streamlit_app.py --server.port 8501 --server.headless true"
echo React + API : http://localhost:8000   (Swagger: http://localhost:8000/docs)
echo Streamlit   : http://localhost:8501
