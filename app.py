from fastapi import FastAPI, UploadFile, File, HTTPException
from deepface import DeepFace
import uuid
import os

app = FastAPI(title="Lightweight Face Recognition API")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/compare")
async def compare_faces(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    try:
        file1_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}.jpg")
        file2_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}.jpg")

        # Save files
        with open(file1_path, "wb") as f:
            f.write(await file1.read())
        with open(file2_path, "wb") as f:
            f.write(await file2.read())

        # Compare faces using DeepFace (uses OpenCV detector by default)
        result = DeepFace.verify(img1_path=file1_path, img2_path=file2_path, enforce_detection=True)

        # Cleanup
        os.remove(file1_path)
        os.remove(file2_path)

        return {
            "matched": result["verified"],
            "confidence": round(result["distance"] * 100, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
