import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not set in .env")

openai_ef = embedding_functions.OpenAIEmbeddingFunction(api_key=api_key)
client = chromadb.Client()
collection = client.get_or_create_collection(
    "resume_bullets",
    embedding_function=openai_ef
)

examples = [
    # ==== IT: Frontend Developer ====
    {"bullet": "Built responsive web applications using React, achieving 99% Lighthouse score.", "skill": "react", "domain": "it", "role": "frontend developer"},
    {"bullet": "Implemented Redux state management for a large-scale e-commerce platform.", "skill": "redux", "domain": "it", "role": "frontend developer"},
    {"bullet": "Refactored CSS and improved cross-browser compatibility, reducing customer support tickets by 20%.", "skill": "css", "domain": "it", "role": "frontend developer"},
    {"bullet": "Developed modular, accessible UI components using Vue.js.", "skill": "vue.js", "domain": "it", "role": "frontend developer"},
    {"bullet": "Optimized web page load times with lazy loading and code splitting.", "skill": "performance optimization", "domain": "it", "role": "frontend developer"},

    # ==== IT: Backend Developer ====
    {"bullet": "Developed RESTful APIs in Node.js, supporting 1M+ monthly users with 99.99% uptime.", "skill": "node.js", "domain": "it", "role": "backend developer"},
    {"bullet": "Implemented OAuth2 authentication and RBAC in a Python Flask application.", "skill": "python", "domain": "it", "role": "backend developer"},
    {"bullet": "Optimized SQL queries, improving database throughput by 60%.", "skill": "sql", "domain": "it", "role": "backend developer"},
    {"bullet": "Designed and maintained scalable microservices architecture with Docker and Kubernetes.", "skill": "docker", "domain": "it", "role": "backend developer"},

    # ==== IT: Fullstack Developer ====
    {"bullet": "Designed and deployed full MERN stack solutions for startups, including CI/CD and cloud hosting.", "skill": "fullstack", "domain": "it", "role": "fullstack developer"},
    {"bullet": "Created a serverless GraphQL API for a SaaS platform, integrating React and AWS Lambda.", "skill": "graphql", "domain": "it", "role": "fullstack developer"},

    # ==== IT: DevOps ====
    {"bullet": "Automated CI/CD pipelines with Jenkins and Docker, enabling daily production deployments.", "skill": "ci/cd", "domain": "it", "role": "devops engineer"},
    {"bullet": "Implemented cloud cost optimization strategies, saving $10K annually on AWS.", "skill": "cloud", "domain": "it", "role": "devops engineer"},

    # ==== IT: Security ====
    {"bullet": "Conducted penetration testing and remediated vulnerabilities, ensuring ISO 27001 compliance.", "skill": "security", "domain": "it", "role": "security engineer"},
    {"bullet": "Developed and enforced security policies for internal and client-facing applications.", "skill": "security policy", "domain": "it", "role": "security analyst"},

    # ==== IT: UI/UX and Web Designers ====
    {"bullet": "Led user research and prototyping to improve mobile app UX, boosting engagement by 30%.", "skill": "ux design", "domain": "it", "role": "ui/ux designer"},
    {"bullet": "Designed custom WordPress themes for 20+ clients, improving website traffic by 45%.", "skill": "wordpress", "domain": "it", "role": "web designer"},
    {"bullet": "Created style guides and wireframes for a fintech dashboard serving 500K+ users.", "skill": "wireframing", "domain": "it", "role": "ui designer"},

    # ==== Data Science & AI ====
    {"bullet": "Built predictive models in Python (scikit-learn), increasing sales forecast accuracy by 18%.", "skill": "machine learning", "domain": "data science", "role": "data scientist"},
    {"bullet": "Deployed NLP pipelines for automated sentiment analysis of customer reviews.", "skill": "nlp", "domain": "data science", "role": "nlp engineer"},
    {"bullet": "Designed interactive dashboards in Tableau for real-time business intelligence.", "skill": "tableau", "domain": "data science", "role": "data analyst"},

    # ==== Healthcare ====
    {"bullet": "Assisted surgeons during 100+ orthopedic procedures, maintaining sterile field and equipment.", "skill": "surgical assistance", "domain": "healthcare", "role": "surgical technologist"},
    {"bullet": "Implemented new triage protocols in ER, reducing patient wait times by 20%.", "skill": "emergency care", "domain": "healthcare", "role": "emergency nurse"},
    {"bullet": "Led digital transformation for patient records, improving retrieval speed by 60%.", "skill": "ehr", "domain": "healthcare", "role": "health it specialist"},
    {"bullet": "Managed outpatient clinic scheduling, raising patient satisfaction to 96%.", "skill": "scheduling", "domain": "healthcare", "role": "clinic manager"},

    # ==== Textile Engineering ====
    {"bullet": "Optimized spinning machine settings, increasing yarn quality by 15%.", "skill": "machine optimization", "domain": "textile", "role": "textile technologist"},
    {"bullet": "Led a project to introduce eco-friendly dyes, reducing wastewater by 40%.", "skill": "sustainable dyeing", "domain": "textile", "role": "textile engineer"},
    {"bullet": "Developed new fabric blends, resulting in a 30% increase in product durability.", "skill": "fabric development", "domain": "textile", "role": "textile engineer"},

    # ==== Leather Technology ====
    {"bullet": "Supervised chrome-free tanning process, cutting chemical costs by 25%.", "skill": "chrome-free tanning", "domain": "leather technology", "role": "leather technologist"},
    {"bullet": "Enhanced finishing techniques, boosting leather softness and customer reviews.", "skill": "finishing", "domain": "leather technology", "role": "leather engineer"},

    # ==== Petroleum Engineering ====
    {"bullet": "Oversaw drilling operations for deepwater wells, increasing yield by 10%.", "skill": "drilling operations", "domain": "petroleum engineering", "role": "drilling engineer"},
    {"bullet": "Managed reservoir simulations to optimize production strategies.", "skill": "reservoir simulation", "domain": "petroleum engineering", "role": "reservoir engineer"},

    # ==== Manufacturing/Production ====
    {"bullet": "Implemented Lean manufacturing principles, reducing waste by 15%.", "skill": "lean manufacturing", "domain": "manufacturing", "role": "production manager"},
    {"bullet": "Supervised 24/7 plant operations, achieving a 98% on-time delivery rate.", "skill": "plant operations", "domain": "manufacturing", "role": "plant supervisor"},

    # ==== Sales and Business ====
    {"bullet": "Surpassed quarterly targets by 35% through strategic client relationships.", "skill": "sales strategy", "domain": "sales", "role": "sales executive"},
    {"bullet": "Increased quarterly revenue by 20% through new partnerships.", "skill": "business development", "domain": "business", "role": "business development manager"},
    {"bullet": "Created and led CRM campaigns, raising customer retention by 28%.", "skill": "crm", "domain": "sales", "role": "crm specialist"},

    # ==== Design ====
    {"bullet": "Created branding and packaging designs adopted by 15+ global brands.", "skill": "branding", "domain": "design", "role": "graphic designer"},
    {"bullet": "Developed interactive prototypes using Figma for enterprise apps.", "skill": "figma", "domain": "design", "role": "ui/ux designer"},

    # ==== Education ====
    {"bullet": "Designed and delivered hybrid curricula for remote STEM learners, improving test scores by 22%.", "skill": "curriculum design", "domain": "education", "role": "curriculum developer"},
    {"bullet": "Implemented digital classroom tools, boosting student engagement by 35%.", "skill": "digital learning", "domain": "education", "role": "teacher"},

    # ==== Customer Service ====
    {"bullet": "Resolved 95% of support tickets within 24 hours, earning 'Top Agent' award.", "skill": "customer service", "domain": "customer service", "role": "customer support specialist"},
    {"bullet": "Trained new hires on CRM systems, reducing onboarding time by 40%.", "skill": "training", "domain": "customer service", "role": "customer service trainer"},

    # ==== Logistics & Supply Chain ====
    {"bullet": "Managed inventory for a distribution center, reducing stockouts by 30%.", "skill": "inventory management", "domain": "logistics", "role": "supply chain analyst"},
    {"bullet": "Optimized delivery routes, cutting transportation costs by 18%.", "skill": "route optimization", "domain": "logistics", "role": "logistics coordinator"},

    # ==== Finance ====
    {"bullet": "Prepared monthly financial reports and reconciled accounts with zero errors.", "skill": "accounting", "domain": "finance", "role": "accountant"},
    {"bullet": "Led cross-team budgeting projects, saving $200K annually.", "skill": "budgeting", "domain": "finance", "role": "financial analyst"},

    # ==== Hospitality ====
    {"bullet": "Coordinated guest services for a 100-room hotel, increasing online reviews to 4.8/5.", "skill": "guest services", "domain": "hospitality", "role": "hotel manager"},
    {"bullet": "Implemented cost control procedures, reducing food waste by 22%.", "skill": "cost control", "domain": "hospitality", "role": "restaurant manager"},

    # ==== Law ====
    {"bullet": "Drafted and reviewed contracts for 25+ corporate clients, ensuring legal compliance.", "skill": "contract drafting", "domain": "law", "role": "legal associate"},
    {"bullet": "Prepared case files and conducted legal research for litigation teams.", "skill": "legal research", "domain": "law", "role": "paralegal"},

    # ==== Agriculture ====
    {"bullet": "Introduced drip irrigation systems, boosting crop yield by 30%.", "skill": "irrigation", "domain": "agriculture", "role": "agricultural engineer"},
    {"bullet": "Managed livestock vaccination programs, reducing disease incidence.", "skill": "veterinary care", "domain": "agriculture", "role": "farm manager"},

    # ==== Government & Public Administration ====
    {"bullet": "Developed and launched community outreach programs, increasing citizen engagement by 50%.", "skill": "community outreach", "domain": "government", "role": "public relations officer"},
    {"bullet": "Coordinated cross-agency emergency response exercises.", "skill": "emergency response", "domain": "government", "role": "emergency planner"},

    # ==== More domains? Just add here! ====
]

documents = [ex["bullet"] for ex in examples]
metadatas = [{"skill": ex["skill"], "domain": ex["domain"], "role": ex["role"]} for ex in examples]
ids = [str(i) for i in range(1, len(examples) + 1)]

collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)
print(f"Seeded ChromaDB with {len(documents)} resume bullets across IT, Healthcare, Design, Engineering, and many more!")
