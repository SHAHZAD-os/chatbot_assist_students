# llm_handler.py

import os
from langchain.llms import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate


os.environ['HUGGINGFACEHUB_API_TOKEN'] = "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" 
token = os.getenv('HUGGINGFACEHUB_API_TOKEN')
if not token:
    print("Hugging Face API token not found!")
else:
    print("Hugging Face API token is set!")

class LLMHandler:
    def __init__(self, model_repo_id="meta-llama/Llama-3.2-11B-Vision-Instruct"):
        self.repo_id = model_repo_id
        self.llm = HuggingFaceEndpoint(
            repo_id=self.repo_id,
            temperature=0.7,  # Specify temperature directly here
            max_length=512,
            token=os.environ['HUGGINGFACEHUB_API_TOKEN']
        )
        self.prompt_template = """
    Context (University Scheme of Studies -):
    - **Minimum Duration**: 4 years (minimum 8 semesters)
    - **Maximum Duration**: 6 years (maximum 12 semesters)
    
    **Course Work Breakdown:**
    - **General Education Courses**: 12 courses, 30 credit hours
    - **Computing Core**: 12 courses, 40 credit hours
    - **Domain Core**: 6 courses, 18 credit hours
    - **Domain Elective**: Choose 7 courses, 21 credit hours
    - **Interdisciplinary/Allied Courses**: 5 courses, 15 credit hours
    - **Field Experience/Internship**: 1 course, 3 credit hours
    - **Capstone Project**: 2 courses, 6 credit hours
    
    **Capstone Project:**
    - CSC498: Final Year Project I, 2 credit hours
    - CSC499: Final Year Project II, 4 credit hours

    **Mandatory Courses Overview:**
    - **Civics and Community Engagement**: HUM208 (2 credit hours)
    - **Entrepreneurship**: MGT250 (2 credit hours)
    - **Expository Writing**: HUM120 (3 credit hours)
    - **Functional English**: HUM104 (3 credit hours)
    - **Ideology and Constitution of Pakistan**: HUM113 (2 credit hours)
    - **Information & Communication Technology Applications**: CSC101 (3 credit hours)
    - **Islamic Studies**: HUM112 (2 credit hours)
    - **Technical and Business Writing**: HUM121 (3 credit hours)
    
    **Core Courses:**
    - **CSC103**: Programming Fundamentals (4 credit hours)
    - **CSC211**: Data Structures (4 credit hours)
    - **CSC241**: Object-Oriented Programming (4 credit hours)
    - **CSC291**: Software Engineering (3 credit hours)
    - **CSC323**: Operating Systems (3 credit hours)

    **Domain Core Courses:**
    - **CSC316**: Advanced Database Systems (3 credit hours)
    - **CSC334**: Parallel and Distributed Computing (3 credit hours)

    **Electives:**
    - **Domain Electives** (Choose 7 courses, 21 credit hours)
    - Some examples: AIC270 (Programming for AI), AIC341 (Computer Vision), CSC303 (Mobile App Development), CSC336 (Web Technologies)

    **Internship:**
    - **CSC395**: Field Experience/Internship (3 credit hours)

    Instructions:
    - If the question relates to specific course details such as prerequisites, credit hours, or course contents, refer to the relevant section from the above scheme.
    - If the information from the provided university scheme of studies is insufficient to answer the question, explicitly mention the lack of detail and suggest where further clarification might be needed (e.g., "Check university-specific resources for more information").

    Question: {question}

    Helpful Answer:
"""

        self.prompt = PromptTemplate.from_template(self.prompt_template)

    def generate_response(self, question, retrieved_info):
        # Prepare the final prompt with formatted text
        final_prompt = self.prompt.format(question=question, retrieved_info=retrieved_info)

        # Call the model with the formatted prompt string
        response = self.llm.invoke(final_prompt)
        return response.strip()

