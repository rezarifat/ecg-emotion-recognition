from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import shutil, os, uuid
from predict import predict_from_file
from report import generate_report

app = FastAPI(title="ECG Emotion Detector")
templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    filename = file.filename.lower()
    if not (filename.endswith('.dcm') or filename.endswith('.csv')):
        return JSONResponse(
            {"error": "Please upload a .dcm (DICOM) or .csv file."},
            status_code=400
        )

    temp_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{file.filename}")
    try:
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        result = predict_from_file(temp_path)
        generate_report(result)
        return JSONResponse(result)

    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": f"Analysis failed: {str(e)}"}, status_code=500)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/report")
async def download_report():
    return FileResponse(
        "report_output.pdf",
        media_type="application/pdf",
        filename="ECG_Emotion_Report.pdf"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)