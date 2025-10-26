"""
Prompt templates.
"""

from langchain_core.prompts import PromptTemplate

PROMPT_TEMPLATE_QUESTIONS = """
You are an expert at creating practice questions based on study material.
Your goal is to prepare a student for their exam. 
You do this by asking questions about the text below:

------------
{text}
------------

Create questions that will prepare the student for their exam.
Focus on the MOST IMPORTANT and FUNDAMENTAL concepts that appear frequently throughout the document.
Prioritize topics that are mentioned multiple times or take up significant space in the material.

Write each question so that it makes complete sense on its own — 
do NOT include phrases like "according to the text", "as mentioned", "in the passage", or similar references.
The questions should sound like they come directly from an exam or quiz, not from a reading comprehension task.

QUESTIONS:
"""
PROMPT_QUESTIONS = PromptTemplate(
    template=PROMPT_TEMPLATE_QUESTIONS,
    input_variables=["text"],
    )

REFINE_TEMPLATE_QUESTIONS = ("""
You are an expert at creating practice questions based on study material.
Your goal is to help a student prepare for an exam.

We have some existing questions: {existing_answer}

Now we have additional context from the document:
------------
{text}
------------

Your task: 
- Update the existing questions to ensure that **all important topics from the entire document** are covered. 
- Do NOT replace questions from previous chunks unless they are redundant. 
- Only add or refine questions to improve coverage. 
- Ensure the final set of questions reflects the full document, not just the most recent chunk.

Prioritize questions about:
- Topics that appear frequently throughout the document
- Fundamental concepts that are central to the subject
- Major themes that span multiple sections
- Core principles that students must understand

Do NOT include phrases like "according to the text", "as mentioned", or "based on the document".
Make the questions fully self-contained and natural, as if they appeared in an exam.

Do not use markdown formatting, bullet points, or any other formatting.
Just numbered questions in plain text.

QUESTIONS:
"""
)
REFINE_PROMPT_QUESTIONS = PromptTemplate(
    input_variables=["existing_answer", "text"],
    template=REFINE_TEMPLATE_QUESTIONS,
)
