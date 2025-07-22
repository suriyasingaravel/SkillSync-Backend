import os
import openai
import re
from dotenv import load_dotenv

# ===== 1. Securely load API keys =====
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set in .env")

# ===== 2. OpenAI and Groq clients =====
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY, base_url="https://api.openai.com/v1")
# groq_client = openai.OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1") # Enable if you use Groq

# ===== 3. LLM & Rule-Based Extraction =====
# You can extend SKILL_DB over time for rule-based fallback:
SKILL_DB = [
    # IT/Tech
    "python", "machine learning", "data analysis", "sql", "excel", "communication",
    "leadership", "project management", "aws", "azure", "docker", "linux", "javascript",
    "react", "vue.js", "node.js", "devops", "security", "ux design", "wordpress", "graphql",
    # Healthcare
    "surgical assistance", "emergency care", "ehr", "scheduling",
    # Textile/Leather/Petroleum
    "machine optimization", "sustainable dyeing", "fabric development", "chrome-free tanning", "drilling operations", "reservoir simulation",
    # Sales, Design, Education, More
    "branding", "crm", "figma", "curriculum design", "digital learning",
    "customer service", "inventory management", "route optimization", "accounting", "budgeting",
    "guest services", "contract drafting", "irrigation", "veterinary care", "community outreach",
    # Expand for your domains!
]

def extract_skills(text, use_llm=True):
    """
    Hybrid: LLM first, fallback to keywords.
    - LLM uses OpenAI to extract skills from ANY domain.
    - If fails, fallback to rule-based on SKILL_DB.
    """
    if use_llm:
        prompt = (
            "Extract a JSON array of all relevant skills, technologies, and competencies from the text below. "
            "Focus on domain-specific, technical, and soft skills. Respond ONLY with a JSON array.\n\n"
            f"Text: {text[:2500]}"
        )
        try:
            result = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_array"}
            )
            import json
            skills = json.loads(result.choices[0].message.content)
            return [skill.lower() for skill in skills if isinstance(skill, str)]
        except Exception as e:
            print(f"LLM skill extraction failed: {e}. Falling back to rule-based.")

    # Rule-based fallback
    text = text.lower()
    found = []
    for skill in SKILL_DB:
        if re.search(rf"\b{re.escape(skill)}\b", text):
            found.append(skill)
    return list(set(found))

def match_skills(jd_skills, resume_skills):
    """
    Find skills present in JD but not in resume.
    """
    return [s for s in jd_skills if s not in resume_skills]

def suggest_rewrites(missing_skills, rag_examples):
    """
    For each missing skill, use RAG if available, else generic advice.
    """
    suggestions = []
    for skill in missing_skills:
        base = rag_examples.get(skill) or f"Add achievements or experience related to {skill}."
        suggestions.append({"skill": skill, "suggestion": base})
    return suggestions

# ====== Advanced: For "Big Project" Structured Prompting ======
def get_structured_resume_feedback(resume_text, jd_text):
    """
    Calls LLM for structured, multi-category feedback (for advanced UI/UX).
    """
    system_prompt = (
        "You are an expert AI career advisor. Analyze the following resume and job description. "
        "Respond in this exact JSON structure: "
        '{"skill_gap_analysis": {"required_skills": [], "present_skills": [], "missing_skills": []}, '
        '"improvement_suggestions": [{"skill": "", "suggestion": ""}], '
        '"formatting_feedback": "", "overall_score": 0, "summary": "", "personalized_roadmap": []}'
    )
    user_prompt = f"Resume:\n{resume_text[:3500]}\n\nJob Description:\n{jd_text[:2000]}"
    try:
        result = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        import json
        feedback = json.loads(result.choices[0].message.content)
        return feedback
    except Exception as e:
        print(f"LLM structured feedback failed: {e}")
        return {"error": str(e)}

# ======================= USAGE EXAMPLE =======================
# In your FastAPI endpoint, you might do:
# feedback = get_structured_resume_feedback(resume_text, jd_text)
# Then merge in RAG suggestions for missing_skills as needed.
