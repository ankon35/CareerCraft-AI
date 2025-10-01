import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate

# Updated imports
from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader, TextLoader


# --------------------------
# Load API Key
# --------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not found in .env file")

# --------------------------
# Initialize Gemini Model
# --------------------------
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)

# --------------------------
# Function to load CV
# --------------------------
def load_cv(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path)
    else:
        raise ValueError("❌ Unsupported file format. Use PDF, DOCX, or TXT.")

    docs = loader.load()
    return " ".join([d.page_content for d in docs])

# --------------------------
# Generate Cover Letter
# --------------------------
def generate_cover_letter(cv_text: str, job_description: str) -> str:
    prompt = ChatPromptTemplate.from_template("""
    You are an expert career coach and professional writer. 
    Using the candidate's CV and the provided Job Description (JD), create a personalized cover letter. 

    CV Content:
    {cv}

    Job Description:
    {jd}

    The cover letter should:
    - Be professional and well-structured
    - Highlight the candidate's most relevant skills and experiences
    - Match the tone of the job description
    - Avoid being too formal or robotic—make it conversational and engaging
    - Stay within one page (approximately 3-4 short paragraphs)

    Now write the cover letter:
    """)

    chain = prompt | model
    response = chain.invoke({"cv": cv_text, "jd": job_description})

    # ✅ FIX: AIMessage object contains .content as string
    return response.content if hasattr(response, "content") else str(response)

# --------------------------
# Main Program
# --------------------------
if __name__ == "__main__":
    cv_path = "Ankon-CV.pdf"  
    job_description = input("📝 Paste Job Description: ").strip()

    print("\n📄 Reading CV...")
    cv_text = load_cv(cv_path)

    print("⚡ Generating personalized cover letter...")
    cover_letter = generate_cover_letter(cv_text, job_description)

    print("\n✅ Generated Cover Letter:\n")
    print(cover_letter)
