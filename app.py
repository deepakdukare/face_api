from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
import uuid
import os

app = FastAPI(title="Lightweight Face Recognition API")

# Allow requests from anywhere (useful for testing with n8n)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Folder to temporarily store uploaded images
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/compare")
async def compare_faces(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    """
    Compare two uploaded face images and return match status and confidence.
    """
    try:
        # Save uploaded files temporarily
        file1_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}.jpg")
        file2_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}.jpg")

        with open(file1_path, "wb") as f:
            f.write(await file1.read())
        with open(file2_path, "wb") as f:
            f.write(await file2.read())

        # Verify faces using DeepFace
        result = DeepFace.verify(img1_path=file1_path, img2_path=file2_path, enforce_detection=True)

        # Clean up temporary files
        os.remove(file1_path)
        os.remove(file2_path)

        # Convert distance to confidence (higher = better match)
        confidence = round((1 - result["distance"]) * 100, 2)

        return {
            "matched": result["verified"],
            "confidence": confidence,
            "distance": result["distance"]
        }

    except ValueError:
        # Face not detected in one or both images
        raise HTTPException(status_code=400, detail="Face not detected in one or both images")
    except Exception as e:
        # Other unexpected errors
        raise HTTPException(status_code=500, detail=str(e))
