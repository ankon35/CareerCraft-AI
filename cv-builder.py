import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate



# --- 1. Load Environment Variables ---
load_dotenv()



# --- 2. Initialize the Language Model (LLM) ---
try:
    # Use a highly capable model like gemini-2.5-flash for quality and speed
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
    print("LLM (Gemini) initialized successfully.")
except Exception as e:
    print(f"Error initializing LLM. Make sure your Google/Gemini API key is set as GOOGLE_API_KEY in your .env file.")
    print(f"Details: {e}")
    exit()





# =========================================================================
# === Task 1: Enhance Professional Summary (Asking for Core Skills) ===
# =========================================================================


summary_template_string = """
You are acting as a veteran career coach. Your goal is to review and rewrite a client's professional summary to make it sound powerful and genuinely human, not like boilerplate AI text.

--- Enhancement Instructions ---
1.  **Integrate Skills:** Naturally weave the provided core skills into the narrative.
2.  **Human Voice:** Rewrite the summary to have a confident, conversational, and non-generic tone, focusing on *impact* and *why* they do the job well.
3.  **Length:** Keep the final summary to a maximum of 3-4 impactful sentences (one paragraph).
4.  **Formatting:** Do not use a header or a title. Just provide the rewritten paragraph.

--- Client's Core Skills ---
{core_skills}

--- Client's Original Summary ---
{original_summary}

--- Your Expertly Rewritten Summary ---
"""
summary_prompt = PromptTemplate(
    template=summary_template_string,
    input_variables=["core_skills", "original_summary"]
)

def task_1_enhance_summary():
    """Runs the enhanced professional summary task."""
    print("\n" + "="*50)
    print("✨ Task 1: Professional Summary Enhancement")
    print("="*50)

    core_skills = input(
        "First, list 3-5 **core skills** you want highlighted (e.g., Data Analysis, Project Management, Client Relations):\n> "
    )
    original_summary = input(
        "\nNow, please paste your current professional summary:\n> "
    )


    
    # Use llm.invoke() directly
    prompt_value = summary_prompt.format_prompt(
        core_skills=core_skills,
        original_summary=original_summary
    )
    
    response = llm.invoke(prompt_value.to_string())


    
    print("\n--- ✅ HERE IS YOUR REVISED PROFESSIONAL SUMMARY ---")
    print("-----------------------------------------------------")
    print(response.content.strip())
    print("-----------------------------------------------------\n")






# =========================================================================
# === Task 2: Enhance Employment History (Bullet Points) ===
# =========================================================================

history_template_string = """
You are a senior copywriter specializing in transforming boring job duties into compelling, achievement-based bullet points that get people noticed. Focus on making the language sound naturally confident and action-oriented, not robotic.

--- Enhancement Instructions ---
1.  **Focus on "I":** Phrase the points as if the candidate is speaking about their **achievement**, using strong, human-sounding action verbs (e.g., I spearheaded, I streamlined, I optimized).
2.  **Show, Don't Tell:** Inject numbers or percentages to quantify impact (e.g., increased revenue by 10%, managed a $50K budget).
3.  **Formatting:** Provide the output as a simple list of bullet points. Do not add any introductory or concluding text.
4.  **Quantity:** Match the number of bullet points provided by the user.

--- User's Original Employment History Details ---
{original_history}

--- Your Expertly Rewritten Bullet Points ---
"""
history_prompt = PromptTemplate(
    template=history_template_string,
    input_variables=["original_history"]
)

def task_2_enhance_history():
    """Runs the enhanced employment history task."""
    print("\n" + "="*50)
    print("💼 Task 2: Employment History Enhancement")
    print("="*50)

    original_history = input(
        "Please enter the bullet points for **one job** (use a hyphen '-' at the start of each point):\n> "
    )


    
    # Use llm.invoke() directly
    prompt_value = history_prompt.format_prompt(original_history=original_history)
    response = llm.invoke(prompt_value.to_string())



    print("\n--- ✅ REVISED EMPLOYMENT HISTORY BULLET POINTS ---")
    print("-----------------------------------------------------")
    print(response.content.strip())
    print("-----------------------------------------------------\n")







# =========================================================================
# === Task 3: Suggest Skills Taglines ===
# =========================================================================

skills_template_string = """
You are an expert ATS (Applicant Tracking System) consultant. Your job is to suggest a set of **Skills Taglines**—keywords that a hiring manager and automated systems would search for—based on the user's specific job title.

--- Suggestion Instructions ---
1.  **Relevance & Variety:** Provide a mix of 6-8 of the most relevant **hard skills, technical tools, and essential soft skills** for this role.
2.  **Tone:** The list should be punchy and professional.
3.  **Format:** Output the skills as a **single, comma-separated list** for easy copying onto a CV. Do not include any other text, quotes, or formatting besides the list itself.

--- User's Job Title ---
{job_title}

--- Suggested Skills Taglines (Comma-separated list) ---
"""
skills_prompt = PromptTemplate(
    template=skills_template_string,
    input_variables=["job_title"]
)

def task_3_suggest_skills():
    """Runs the suggested skills taglines task."""
    print("\n" + "="*50)
    print("💡 Task 3: Skills Taglines Suggestion")
    print("="*50)

    job_title = input("What is the exact **JOB TITLE** you are applying for (e.g., Senior Software Engineer, Marketing Specialist):\n> ")

    print("\nAnalyzing keywords for ATS compatibility...")
    
    # Use llm.invoke() directly
    prompt_value = skills_prompt.format_prompt(job_title=job_title)
    response = llm.invoke(prompt_value.to_string())

    print("\n--- ✅ SUGGESTED CORE SKILLS TAGLINES (ATS Optimized) ---")
    print("---------------------------------------------------------")
    print(response.content.strip())
    print("---------------------------------------------------------\n")







# =========================================================================
# === Main Menu (Test Mode) ===
# =========================================================================

def main_menu():
    """Presents an interactive menu to run tasks."""
    while True:
        print("\n" + "="*50)
        print("          CV ENHANCEMENT MENU (TEST MODE)")
        print("="*50)
        print("1. Enhance Professional Summary (Task 1)")
        print("2. Enhance Employment History (Task 2)")
        print("3. Suggest Skills Taglines (Task 3)")
        print("4. Exit")
        print("="*50)

        choice = input("Enter your choice (1-4): ")

        if choice == '1':
            task_1_enhance_summary()
        elif choice == '2':
            task_2_enhance_history()
        elif choice == '3':
            task_3_suggest_skills()
        elif choice == '4':
            print("Exiting CV Enhancer. Good luck with your job search!")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 4.")





if __name__ == "__main__":
    main_menu()