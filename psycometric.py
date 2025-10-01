import os
import re
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini LLM
chat = ChatGoogleGenerativeAI(api_key=gemini_api_key, model="gemini-2.5-flash", temperature=0.3)

# Step 1: Generate 15 psychometric questions
prompt_generate = PromptTemplate(
    input_variables=[],
    template="""
Generate 15 psychometric test questions covering:
- Situational judgment
- Numerical reasoning
- Verbal reasoning
- Non-verbal reasoning

Each question should have:
- The question text
- 4 options (labeled 1-4)
- The correct answer number (1-4)

Format as plain text like:
Question: ...
1. Option
2. Option
3. Option
4. Option
Answer: 2

Separate each question with a blank line.
"""
)

questions_text = chat.predict(prompt_generate.format())

# Step 2: Parse questions
questions = []
for block in questions_text.strip().split("\n\n"):
    lines = block.strip().split("\n")
    if len(lines) < 6:
        continue
    question_text = lines[0].replace("Question: ", "").strip()
    options = [lines[1][3:].strip(), lines[2][3:].strip(), lines[3][3:].strip(), lines[4][3:].strip()]
    
    # Extract first digit from the answer line
    match = re.search(r'\d', lines[5])
    if match:
        answer = int(match.group())
    else:
        answer = 1  # fallback
    questions.append({"question": question_text, "options": options, "answer": answer})

# Step 3: User attempts questions one by one
user_answers = []
for i, q in enumerate(questions, 1):
    print(f"\nQuestion {i}: {q['question']}")
    for idx, option in enumerate(q['options'], 1):
        print(f"{idx}. {option}")
    
    while True:
        user_ans = input("Your answer (1-4): ").strip()
        if user_ans in ["1", "2", "3", "4"]:
            user_answers.append(int(user_ans))
            break
        else:
            print("Invalid input! Enter a number between 1 and 4.")

# Step 4: Display test results
correct_count = 0
print("\n--- Test Results ---\n")
for i, q in enumerate(questions):
    print(f"Question {i+1}: {q['question']}")
    print(f"Your answer: {user_answers[i]} - {q['options'][user_answers[i]-1]}")
    print(f"Correct answer: {q['answer']} - {q['options'][q['answer']-1]}\n")
    if user_answers[i] == q['answer']:
        correct_count += 1

wrong_count = len(questions) - correct_count
print(f"Total Questions: {len(questions)}")
print(f"Correct Answers: {correct_count}")
print(f"Wrong Answers: {wrong_count}")

print("\n--- Waiting for the AI report.... ---\n")

# Step 5: Prepare feedback
feedback_data = []
for i, q in enumerate(questions):
    feedback_data.append(f"Q{i+1}: {q['question']}\nYour answer: {user_answers[i]}\nCorrect answer: {q['answer']}\n")

feedback_prompt = PromptTemplate(
    input_variables=["feedback_data", "correct_count", "wrong_count"],
    template="""
The user has completed a psychometric test with 15 questions. Here are the results:

- Correct Answers: {correct_count}
- Wrong Answers: {wrong_count}

Below are the user's responses compared to the correct answers:

{feedback_data}

Provide a concise and clear summary of the results. Include:
1. **Strengths**: Briefly highlight areas where the user performed well (correct answers).
2. **Areas for Improvement**: Focus on the questions where the user answered incorrectly, suggesting how to improve.
3. **Overall Assessment**: Give a simple evaluation of the user's performance, pointing out potential strengths and areas of improvement in psychometric reasoning abilities.

Ensure the feedback is actionable and encouraging, with specific suggestions for improvement.
"""
)

feedback = chat.predict(feedback_prompt.format(
    feedback_data="\n".join(feedback_data),
    correct_count=correct_count,
    wrong_count=wrong_count
))

print("\n--- Personalized Feedback ---\n")
print(feedback)
