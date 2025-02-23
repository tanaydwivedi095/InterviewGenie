import random
import torch
from openai import OpenAI
class OpenAI_QuestionGenerator:
    def __init__(self, openai_api_key):
        self.openai_api_key = openai_api_key
        self.number_of_questions = 5

    def load_llm_model(self):
        llm_model = OpenAI(api_key = self.openai_api_key)
        return llm_model

    def generate_prompt(self, difficulty):
        question_type = {
            0: "definition",
            1: "formula",
            2: "scenario",
            3: "case-study",
            4: "algorithmic theory",
            5: "mathematics",
            6: "probability and statistics",
            7: "real life application",
        }
        question_types = list()
        for i in range(self.number_of_questions):
            random_idx = random.randint(0, len(question_type)-1)
            question_types.append(f"{i+1}. {question_type[random_idx]}")
        question_types = "\n".join(question_types)
        print(self.number_of_questions)
        base_prompt = f"""
        Generate {self.number_of_questions} questions for an Interview for the position of Machine Learning Engineer.
        Each question should be of type as mentioned below:
        {question_types}
        The output provided should be in format as given below:
        Question 1: Question Type 1 ....
        Question 2: Question Type 2 ....
        Question 3: Question Type 3 .....
        and so on.
        No extra text should be generated as answer.
        The case study questions should be well defined.
        The questions should only be of the specified type.
        Do not mention the type of question in the generated question.
        Make the questions hard on a scale of {difficulty} out of 10.
        """
        base_prompt = base_prompt.format(question_types = question_types, 
                                         number_of_questions = self.number_of_questions,
                                         difficulty=difficulty)
        dialogue_template = [{
            "role":"user",
            "content":base_prompt
        }]
        return dialogue_template

    def generate_questions(self, llm_model, prompt):
        questions = llm_model.chat.completions.create(
            model="o1-mini",
            store=False,
            messages=prompt
        )
        return questions

    def post_process_questions(self, questions):
        questions = questions.choices[0].message.content.split("Question")
        questions = [f"{question[3:].strip().replace('\n', "")}" for question in questions if question]
        return questions
