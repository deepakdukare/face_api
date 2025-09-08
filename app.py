from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
import uuid
import os
import shutil

app = FastAPI(title="Lightweight Face Recognition API")

# Allow requests from anywhere (useful for testing with n8n / Postman)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Temporary upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.post("/compare")
async def compare_faces(
    data: UploadFile = File(..., description="First face image"),
    data2: UploadFile = File(..., description="Second face image")
):
    """
    Compare two uploaded face images and return match status, confidence, and distance.
    """
    file1_path, file2_path = None, None
    try:
        # Ensure both files are provided
        if not data or not data2:
            raise HTTPException(status_code=400, detail="Both data and data2 files are required")

        # Validate content type
        allowed_types = ["image/jpeg", "image/png"]
        if data.content_type not in allowed_types or data2.content_type not in allowed_types:
            raise HTTPException(status_code=400, detail="Only JPEG and PNG images are supported")

        # Save uploaded files temporarily
        file1_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}.jpg")
        file2_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}.jpg")

        with open(file1_path, "wb") as f1:
            shutil.copyfileobj(data.file, f1)
        with open(file2_path, "wb") as f2:
            shutil.copyfileobj(data2.file, f2)

        # Run DeepFace verification
        result = DeepFace.verify(
            img1_path=file1_path,
            img2_path=file2_path,
            enforce_detection=True
        )

        # Confidence (higher = stronger match)
        confidence = round((1 - result["distance"]) * 100, 2)

        return {
            "matched": result["verified"],
            "confidence": confidence,
            "distance": result["distance"]
        }

    except ValueError:
        # Face not detected
        raise HTTPException(status_code=400, detail="Face not detected in one or both images")
    except Exception as e:
        print(f"❌ Internal Error: {e}")  # log error
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
    finally:
        # Cleanup files if they exist
        if file1_path and os.path.exists(file1_path):
            os.remove(file1_path)
        if file2_path and os.path.exists(file2_path):
            os.remove(file2_path)
