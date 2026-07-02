import io
import json
import time
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel, Field
from pypdf import PdfReader
from google import genai
from google.genai import types

# Initialize the Gemini Client
# Note: Never hardcode your API keys in production scripts. Use environment variables.
client = genai.Client(api_key="AQ.Ab8RN6IdB1ECnolWPaH4b3g7AnDn7JXV4okl79n0nIVfVixYTw")

app = FastAPI(title="Resume Parser & Pipeline API")

# --- Structured Output Schema ---
class ResumeExtraction(BaseModel):
    hard_skills: list[str] = Field(description="List of technical, software, or domain-specific hard skills.")
    soft_skills: list[str] = Field(description="List of interpersonal, leadership, or communication soft skills.")
    experience_years: int = Field(description="Total years of professional experience calculated from the timeline.")

@app.post("/extract-resume/", response_model=ResumeExtraction)
async def extract_resume(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    try:
        # 1. Read and parse the PDF
        pdf_bytes = await file.read()
        pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
        
        extracted_text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                extracted_text += page_text + "\n"
                
        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract any text from the PDF.")

        # 2. Extract Data using Gemini Structured Outputs
        prompt = f"You are an expert technical recruiter. Extract the candidate's skills and total years of experience from this resume text:\n\n{extracted_text}"
        
        response = client.models.generate_content(
            model='gemini-3.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ResumeExtraction,
                temperature=0.1, # Low temperature for factual extraction
            ),
        )

        # 3. Parse and return the structured data
        parsed_data = ResumeExtraction.model_validate_json(response.text)
        return parsed_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
