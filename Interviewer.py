from Gemma_QuestionGenerator import Gemma_QuestionGenerator
from OpenAI_QuestionGenerator import OpenAI_QuestionGenerator
from Interviewee import Interviewee
import random
from openai import OpenAI

class Interviewer:

    def __init__(self, name, generator_type, api_key=None, difficulty=None):
        self.name = name
        self.generator_type = generator_type
        self.api_key = api_key
        self.difficulty = difficulty

    def generate_questions(self):
        if self.generator_type.lower() == "free":
            generator = Gemma_QuestionGenerator()
            llm_model = generator.load_llm_model()
            tokenizer = generator.load_tokenizer()
            prompt = generator.generate_prompt(tokenizer)
            output_text = generator.generate_questions(llm_model, tokenizer, prompt)
            questions = generator.post_process_questions(output_text)
            return questions
        else:
            generator = OpenAI_QuestionGenerator(self.api_key)
            llm_model = generator.load_llm_model()
            prompt = generator.generate_prompt(self.difficulty)
            output_text = generator.generate_questions(llm_model, prompt)
            questions = generator.post_process_questions(output_text)
            return questions

    def get_a_question(self, questions):
        question = questions[0]
        if len(questions)>1:
            questions = questions[1:]
        else:
            questions = []
        return questions, question

    def score_answer(self, query, user_answer):
        
        new_prompt = f"""
            I have a question and an answer, I need you to rate the answer on a scale of 0 to 10,
            0 being the worst score and 10 being the best score.
            The question was {query}.
            The answer is {user_answer}.
            Score it from a perspective of an Interviewer of Machine Learning Engineer job post.
            I need the answer in a specific format that is 'score'
            I need no extra character to be generated.
        """

        dialogue_template = [{
            "role":"user",
            "content": new_prompt
        }]

        llm_model = OpenAI(api_key=self.api_key)
        score = llm_model.chat.completions.create(
            model="gpt-4o-mini",
            store=False,
            messages=dialogue_template
        )
        score = score.choices[0].message.content
        return score