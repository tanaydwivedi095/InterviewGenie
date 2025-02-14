import streamlit as st
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import re

EMBEDDING_MODEL = "all-mpnet-base-v2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K = 10
LLM_MODEL = "google/gemma-2b-it"
TEMPERATURE = 0.3
MAX_NEW_TOKENS = 2056
    
def ask(query):
    flat_embeddings = pd.read_csv("embeddings_v4.csv").to_numpy()
    flat_data = pd.read_csv("data_v4.csv")["0"].tolist()
    pages_and_metadata_embeddings = np.array(flat_embeddings, dtype=np.float32)
    pages_and_metadata_embeddings = torch.tensor(pages_and_metadata_embeddings, dtype=torch.float32).to(device)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    query_embeddings = embedding_model.encode(query, convert_to_tensor=True).to(device)
    dot_score = util.dot_score(query_embeddings, pages_and_metadata_embeddings)[0]
    top_scores, top_indices = torch.topk(dot_score, K)
    context = list()
    for idx in top_indices:
        context.append(flat_data[idx.item()])
    model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=LLM_MODEL,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=False,
            ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
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
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**input_ids, temperature=TEMPERATURE, do_sample=True, max_new_tokens=MAX_NEW_TOKENS)
    output_text = tokenizer.decode(outputs[0])
    return output_text

def clean_output_text(output_text):
    idx = output_text.find("Answer")
    print(output_text)
    answer = output_text[idx + 7:]
    answer = answer.replace("**", "")
    answer = answer.replace("<start_of_turn>model", "")
    answer = re.sub("<.*?>", "", answer)
    return answer


st.title("INTERVIEW BOT")

query = st.text_input("Enter your question:")
print(query)

if(len(query) > 0):
    output_text = ask(query)
    st.write("Answer")
    clean_text = clean_output_text(output_text)
    st.write(clean_text)