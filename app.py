import os
import json
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from utils.pdf_parser import extract_text_from_pdf
from utils.rag import get_rag_examples
from openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

STRUCTURED_PROMPT = """
You are an expert career advisor for all domains.
Given the following resume and job description, analyze them and respond in this exact JSON format:
{{
  "skill_gap_analysis": {{
    "required_skills": [],
    "present_skills": [],
    "missing_skills": []
  }},
  "improvement_suggestions": [
    {{"skill": "", "suggestion": ""}}
  ],
  "formatting_feedback": "",
  "overall_score": 0,
  "summary": "",
  "personalized_roadmap": []
}}

Resume:
{resume_text}

Job Description:
{jd_text}
"""

@app.post("/analyze/")
async def analyze(
    resume: UploadFile = File(...),
    jd_text: str = Form(...)
):
    if resume.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Please upload a PDF for the resume.")
    resume_bytes = await resume.read()
    resume_text = extract_text_from_pdf(resume_bytes)
    jd_text = jd_text.strip()

    prompt = STRUCTURED_PROMPT.format(
        resume_text=resume_text[:3500],  # (truncated for token safety)
        jd_text=jd_text[:2000]
    )

    response = client.chat.completions.create(
        model="gpt-4o",  # Or another OpenAI model
        messages=[
            {"role": "system", "content": "You are an expert career advisor."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        response_format={"type": "json_object"}
    )
    ai_json = response.choices[0].message.content
    try:
        data = json.loads(ai_json)
    except Exception:
        data = {"error": "AI output could not be parsed. Output was:", "raw": ai_json}

    # Optionally, augment with your RAG suggestions for missing_skills:
    rag_examples = get_rag_examples(data.get("skill_gap_analysis", {}).get("missing_skills", []))
    for suggestion in data.get("improvement_suggestions", []):
        skill = suggestion.get("skill")
        if skill in rag_examples and rag_examples[skill]:
            suggestion["rag_example"] = rag_examples[skill]

    return data
