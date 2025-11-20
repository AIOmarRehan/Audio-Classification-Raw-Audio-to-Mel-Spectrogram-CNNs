from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tempfile, shutil
from app.preprocess import preprocess_audio
from app.model import predict
import numpy as np

app = FastAPI(title="General Audio Classifier")

@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        # Preprocess → multiple spectrograms
        imgs = preprocess_audio(tmp_path)

        # Predict all chunks
        all_preds = []
        all_confidences = []
        for img in imgs:
            label, confidence, probs = predict(img)
            all_preds.append(label)
            all_confidences.append(confidence)

        # Combine predictions (majority vote with confidence tiebreaker)
        from collections import Counter, defaultdict
        counter = Counter(all_preds)
        max_count = max(counter.values())
        candidates = [k for k, v in counter.items() if v == max_count]

        if len(candidates) == 1:
            final_label = candidates[0]
        else:
            # Tie-breaker using sum of confidences
            confidence_sums = defaultdict(float)
            for i, label in enumerate(all_preds):
                if label in candidates:
                    confidence_sums[label] += all_confidences[i]
            final_label = max(confidence_sums, key=confidence_sums.get)

        # Average confidence for final label
        final_confidence = np.mean([all_confidences[i] for i, label in enumerate(all_preds) if label == final_label])

        return JSONResponse(content={
            "predicted_label": final_label,
            "confidence": round(final_confidence, 3),
            "all_predictions": all_preds,
            "all_confidences": [round(c,3) for c in all_confidences]
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
