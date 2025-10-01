import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader


# ---------------------------
# Load API Key
# ---------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("Please set GEMINI_API_KEY in your .env file.")


# ---------------------------
# CV Ranking Function
# ---------------------------
def analyze_cv(cv_file_path, job_description):
    # Load CV directly as a Document
    if cv_file_path.endswith(".pdf"):
        loader = PyPDFLoader(cv_file_path)
    elif cv_file_path.endswith(".docx"):
        loader = UnstructuredWordDocumentLoader(cv_file_path)
    else:
        raise ValueError("Only PDF and DOCX files are supported.")
    
    documents = loader.load()
    cv_text = "\n".join([doc.page_content for doc in documents])

    # Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0.3
    )

    # Prompt Template
    template = """
You are an expert career consultant and recruiter with years of experience evaluating CVs. 
Your task is to provide a personalized and in-depth analysis of the following CV based on the job description provided.

Please analyze the CV carefully and provide a detailed, thoughtful, and personalized response addressing the following aspects:

1. **Relevance Score (0-100)**: 
   - How well does this CV align with the requirements of the job description? 
   - Rate the relevance of the candidate’s experience, skills, and achievements in relation to the job’s expectations.

2. **Content Quality and Clarity**:
   - **Clarity of Communication**: Is the CV easy to read and understand? Does it communicate the candidate’s qualifications and experiences clearly? 
   - **Focus on Achievements**: Does the CV effectively showcase accomplishments and measurable impact, rather than just listing job duties?
   - Provide personalized suggestions for improving the CV’s narrative.

3. **Grammar, Spelling, and Language Use**:
   - Assess the overall writing quality. Is the language professional, error-free, and suitable for the job?

4. **Formatting and Visual Appeal**:
   - How well is the CV organized? Does it use clear sections, appropriate fonts, and consistent styling?

5. **Keyword and Skill Matching**:
   - Analyze how well the CV includes the specific skills and qualifications mentioned in the job description.

6. **Strengths**:
   - Identify 5 standout qualities of the CV, such as well-demonstrated skills, experience, or accomplishments.

7. **Weaknesses and Areas for Improvement**:
   - Point out any gaps, missing information, or areas where the CV could be strengthened.

8. **Personalized Feedback for Improvement**:
   - Provide 5 detailed suggestions on how to make the CV more compelling based on the specific job description and your expert understanding.

9. **Final Recommendation**:
   - Based on the overall analysis, provide a final recommendation on how the CV can be further refined to improve the candidate's chances of getting hired for the specific job.
  
**CV Text:**
{cv_text}

**Job Description:**
{job_description}

Please present the results in the following format:

**Score**: <number>/100  
**Strengths**:  
- <list of strengths>  
**Weaknesses**:  
- <list of weaknesses>  
**Personalized Feedback**:  
- <detailed feedback with specific suggestions, explanations, and actionable recommendations for improvement>
"""

    # Prepare the prompt
    prompt = PromptTemplate(
        input_variables=["cv_text", "job_description"],
        template=template
    )
    final_prompt = prompt.format(cv_text=cv_text, job_description=job_description)

    # Invoke LLM
    response = llm.invoke([HumanMessage(content=final_prompt)])
    result_content = response.content.strip()

    # Parsing the response into structured format (score, strengths, weaknesses, feedback)
    def parse_response(response_text):
        # Extracting the content using simple pattern matching or regex (could be more sophisticated)
        score = None
        strengths = []
        weaknesses = []
        feedback = []

        # Score extraction
        if "Score" in response_text:
            score_line = response_text.split("Score")[1].split("\n")[0].strip()
            score = score_line.split(":")[1].strip() if score_line else "N/A"

        # Extracting strengths
        if "Strengths" in response_text:
            strengths_section = response_text.split("Strengths")[1].split("Weaknesses")[0].strip()
            strengths = [s.strip() for s in strengths_section.split("\n") if s.strip()]

        # Extracting weaknesses
        if "Weaknesses" in response_text:
            weaknesses_section = response_text.split("Weaknesses")[1].split("Personalized Feedback")[0].strip()
            weaknesses = [w.strip() for w in weaknesses_section.split("\n") if w.strip()]

        # Extracting personalized feedback
        if "Personalized Feedback" in response_text:
            feedback_section = response_text.split("Personalized Feedback")[1].split("Final Recommendation")[0].strip()
            feedback = [f.strip() for f in feedback_section.split("\n") if f.strip()]

        return {
            "Score": score,
            "Strengths": strengths[:5],  # Max 5 strengths
            "Weaknesses": weaknesses[:5],  # Max 5 weaknesses
            "Personalized Feedback": feedback[:5]  # Max 5 feedback points
        }

    # Parse the response content into structured format
    result = parse_response(result_content)
    
    return result


# ---------------------------
# Example Run
# ---------------------------
if __name__ == "__main__":
    cv_file = "Ankon-CV.pdf"  # or sample_cv.docx
    job_description = """
    We are looking for a Data Analyst with experience in SQL, Python, Excel, and Power BI.
    Candidate should have strong analytical skills, experience with dashboards, and business communication skills.
    """

    result = analyze_cv(cv_file, job_description)
    
    # Output the formatted result
    print(f"Score: {result['Score']}")
    print("Strengths:")
    for strength in result['Strengths']:
        print(f"- {strength}")
    print("Weaknesses:")
    for weakness in result['Weaknesses']:
        print(f"- {weakness}")
    print("Personalized Feedback:")
    for feedback_point in result['Personalized Feedback']:
        print(f"- {feedback_point}")
