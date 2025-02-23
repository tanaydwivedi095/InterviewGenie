import random
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

class Gemma_QuestionGenerator:

    def __init__(self):
        self.llm_model_name = "google/gemma-2b-it"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.number_of_questions = 5
        self.question_type = {
            0: "definition",
            1: "formula",
            2: "scenario",
            3: "case-study",
            4: "algorithmic theory",
            5: "mathematics",
            6: "probability and statistics",
        }
        self.temperature = 0.3
        self.max_new_tokens = 2048

    def load_llm_model(self):
        llm_model = AutoModelForCausalLM.from_pretrained(self.llm_model_name,
                                                        torch_dtype=torch.float16)
        llm_model = llm_model.to(self.device)
        return llm_model

    def load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
        return tokenizer

    def generate_prompt(self, tokenizer):
        question_types = list()
        for i in range(self.number_of_questions):
            random_idx = random.randint(0, len(self.question_type)-1)
            question_types.append(f"{i+1}. {self.question_type[random_idx]}")
        question_types = "\n".join(question_types)
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
        """
        base_prompt = base_prompt.format(question_types = question_types, 
                                         number_of_questions = self.number_of_questions)
        dialogue_template = [{
            "role":"user",
            "message":base_prompt
        }]
        prompt = tokenizer.apply_chat_template(conversation=dialogue_template,
                                               tokenize=False,
                                               add_generation_prompt=True)
        return base_prompt
        
    def generate_questions(self, llm_model, tokenizer, prompt):
       input_ids = tokenizer([prompt],
                             return_tensors="pt")
       input_ids = input_ids.to(self.device)
       outputs = llm_model.generate(**input_ids,
                                    temperature=self.temperature,
                                    do_sample=True,
                                    max_new_tokens=self.max_new_tokens)
       output_text = tokenizer.batch_decode(outputs)[0]
       return output_text

    def post_process_questions(self, questions):
        questions = questions[questions.find("\n\n**")+4:].replace("**","").replace("\n","")
        questions = questions.split("Question")
        questions = [question for question in questions if question]
        questions[-1] = questions[-1][:-5]
        return questions