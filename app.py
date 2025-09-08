from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
import uuid
import os
import shutil

app = FastAPI(title="Lightweight Face Recognition API")

# Allow requests from anywhere (useful for n8n / frontend integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Temporary folder for uploads
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.post("/compare")
async def compare_faces(
    data: UploadFile = File(...), 
    data2: UploadFile = File(...)
):
    """
    Compare two uploaded face images and return match status + confidence.
    """
    file1_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}.jpg")
    file2_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}.jpg")

    try:
        # Save uploaded files safely
        with open(file1_path, "wb") as buffer:
            shutil.copyfileobj(data.file, buffer)
        with open(file2_path, "wb") as buffer:
            shutil.copyfileobj(data2.file, buffer)

        # Run DeepFace verification
        result = DeepFace.verify(
            img1_path=file1_path, 
            img2_path=file2_path, 
            enforce_detection=True,
            model_name="VGG-Face"  # ✅ lightweight model, faster + less memory
        )

        # Confidence calculation
        confidence = round((1 - result.get("distance", 0)) * 100, 2)

        return {
            "matched": result.get("verified", False),
            "confidence": confidence,
            "distance": result.get("distance", None),
        }

    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Face not detected in one or both images"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
    finally:
        # Clean up files to save memory
        for f in [file1_path, file2_path]:
            if os.path.exists(f):
                os.remove(f)
