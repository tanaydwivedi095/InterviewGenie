import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import re

class Interviewee:
    
    def __init__(self, name):
        self.name = name
        self.llm_model_name = "google/gemma-2b-it"
        self.embedding_model_name = "all-MiniLM-L12-v2"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.k = 20
        self.temperature = 0.2
        self.max_new_tokens = 2048
        self.embedding_file_name = "embeddings_v5.csv"
        self.data_file_name = "data_v5.csv"
    
    def load_embeddings(self):
        embeddings = pd.read_csv(self.embedding_file_name).to_numpy()
        embeddings = np.array(embeddings, dtype=np.float32)
        embeddings = torch.tensor(embeddings, dtype=torch.float32)
        embeddings = embeddings.to(self.device)
        return embeddings
    
    def load_data(self):
        data = pd.read_csv(self.data_file_name)["0"].tolist()
        return data

    def load_embedding_model(self):
        model = SentenceTransformer(self.embedding_model_name)
        return model

    def embed_query(self, query, model):
        embedding = model.encode(query, convert_to_tensor=True)
        embedding = embedding.to(self.device)
        return embedding

    def get_top_k_similar_context(self, query_embedding, embeddings, data):
        dot_score = util.dot_score(query_embedding, embeddings)[0]
        top_scores, top_indices = torch.topk(dot_score, self.k)
        context = list()
        for idx in top_indices:
            context.append(data[idx.item()])
        return context

    def load_llm_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path = self.llm_model_name,
            torch_dtype = torch.float16,
            low_cpu_mem_usage = False
        )
        model = model.to(self.device)
        return model

    def load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
        return tokenizer

    def generate_prompt(self, tokenizer, context, query):
        context = "\n -".join(context)
        base_prompt = f'''Bases on the following context items, please answer the query
        Context Items:
        {context}
        Query:
        {query}
        Answer:'''
        base_prompt = base_prompt.format(context=context, query=query)
        dialogue_template = [{
            "role": "user",
            "content": base_prompt,
        }]
        prompt = tokenizer.apply_chat_template(conversation=dialogue_template,
                                               tokenize=False,
                                               add_generation_prompt=True)
        return prompt

    def generate_answer(self, tokenizer, llm_model, prompt):
        input_ids = tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = llm_model.generate(**input_ids, temperature=self.temperature, do_sample=True, max_new_tokens=self.max_new_tokens)
        output_text = tokenizer.decode(outputs[0])
        return output_text

    def clean_output_text(self, output_text):
        idx = output_text.find("Answer")
        answer = output_text[idx + 7:]
        answer = answer.replace("**", "")
        answer = answer.replace("<start_of_turn>model", "")
        answer = re.sub("<.*?>", "", answer)
        return answer
