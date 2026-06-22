import os
import base64
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from pymongo import MongoClient
import uvicorn

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# ───────────── LOAD ENV ─────────────
load_dotenv()

app = FastAPI(title="AI Nutrition Analyzer API", version="1.0")


@app.get("/")
async def root():
    # Redirect visitors of the backend root to the frontend (update URL once deployed)
    return RedirectResponse(url="https://your-frontend-url.onrender.com")


# ───────────── DATABASE ─────────────
client = MongoClient(os.getenv("MONGO_URL"))
db = client[os.getenv("DB_NAME")]
analyses_collection = db["analyses"]

# ───────────── GEMINI LLM ─────────────
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY")
)


# ───────────── MODELS ─────────────
class AnalysisUpdate(BaseModel):
    notes: Optional[str] = None
    status: Optional[str] = None


# ───────────── HELPERS ─────────────
def serialize_analysis(analysis):
    if not analysis:
        return None
    analysis.pop("_id", None)
    return analysis


def get_next_analysis_id():
    existing_ids = set(
        a["id"] for a in analyses_collection.find({}, {"id": 1})
    )
    if not existing_ids:
        return 1
    max_id = max(existing_ids)
    for candidate in range(1, max_id + 2):
        if candidate not in existing_ids:
            return candidate


def encode_image(image_content: bytes) -> str:
    return base64.b64encode(image_content).decode()


def analyze_food_image(image_bytes: bytes, content_type: str) -> dict:
    image_b64 = encode_image(image_bytes)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a nutrition expert capable of analyzing food images and providing detailed nutritional advice."),
        ("human", [
            {
                "type": "text",
                "text": """Analyze the image and provide a comprehensive nutritional breakdown and health advice. Follow these steps:
                1. Identify each distinct food/drink item visible in the image.
                2. Estimate the portion size for each item (e.g., grams, cups, pieces).
                3. Estimate calories, protein, carbohydrates, fat, and fiber for each item.
                4. Sum these into total values for the full meal.
                5. Give a brief, balanced health note (e.g., sodium/sugar content, missing food groups) — framed as general nutrition information, not personalized medical advice.
                6. Return the result in JSON format with keys: items, totals, notes."""
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:{content_type};base64,{image_b64}"}
            }
        ])
    ])

    chain = prompt | llm | JsonOutputParser()
    return chain.invoke({})


# ───────────── ANALYZE MEAL ─────────────
@app.post("/analyze")
async def analyze_meal(file: UploadFile = File(...)):
    content_type = file.content_type
    file_bytes = await file.read()

    if len(file_bytes) > 10_000_000:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB")

    try:
        result = analyze_food_image(file_bytes, content_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    new_analysis = {
        "id": get_next_analysis_id(),
        "filename": file.filename,
        "items": result.get("items", []),
        "totals": result.get("totals", {}),
        "notes": result.get("notes", ""),
        "status": "completed",
        "created_at": datetime.utcnow().isoformat()
    }
    analyses_collection.insert_one(new_analysis)
    new_analysis.pop("_id", None)
    return new_analysis


# ───────────── GET ALL ANALYSES ─────────────
@app.get("/analyses")
async def get_analyses(status_filter: Optional[str] = None):
    query = {}
    if status_filter:
        query["status"] = status_filter

    analyses = []
    for a in analyses_collection.find(query).sort("id", -1):
        analyses.append(serialize_analysis(a))
    return analyses


# ───────────── GET ANALYSIS BY ID ─────────────
@app.get("/analyses/{analysis_id}")
async def get_analysis(analysis_id: int):
    analysis = analyses_collection.find_one({"id": analysis_id})
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return serialize_analysis(analysis)


# ───────────── UPDATE ANALYSIS ─────────────
@app.put("/analyses/{analysis_id}")
async def update_analysis(analysis_id: int, data: AnalysisUpdate):
    update_data = {k: v for k, v in data.dict().items() if v is not None}
    if not update_data:
        raise HTTPException(status_code=400, detail="No fields to update")

    result = analyses_collection.update_one(
        {"id": analysis_id},
        {"$set": update_data}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return {"message": "Updated successfully"}


# ───────────── DELETE ANALYSIS ─────────────
@app.delete("/analyses/{analysis_id}")
async def delete_analysis(analysis_id: int):
    result = analyses_collection.delete_one({"id": analysis_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return {"message": "Deleted successfully"}


# ───────────── STATS ─────────────
@app.get("/stats")
async def get_stats():
    total = analyses_collection.count_documents({})
    completed = analyses_collection.count_documents({"status": "completed"})

    return {
        "total_analyses": total,
        "completed": completed
    }


# ───────────── RUN ─────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
