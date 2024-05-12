# build a simple web app where you can ask it questions

import streamlit as st
#import rag_langchain_helper
import transformers
import torch
from transformers import AutoModel, AutoTokenizer
from pymilvus import MilvusClient, DataType
import os
import requests

def chunk_text(text, max_length, tokenizer, model):
    # Tokenize the text
    #tokens = tokenizer.tokenize(text,return_tensors="pt")
    tokens = tokenizer.encode_plus(text,return_tensors="pt")
    # Split the tokens into chunks of max_length
    chunks = [tokens.input_ids[:, i:i+max_length//2] for i in range(0, (tokens.input_ids.shape[1]), max_length//2)]
    return chunks

def query_vect_db(content, tokenizer, model):
    chunks = chunk_text(content, 512, tokenizer, model)
    for chunk in chunks:
        embeddings = model(chunk, return_dict=True).last_hidden_state[:, 0]
        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)

    print("embedded")
    # 1. Set up a Milvus client
    client = MilvusClient(
        uri="http://localhost:19530"
    )  
    
    res = client.search(
        collection_name="llama2",     # target collection
        data=[list(embeddings.flatten())],                # query vectors
        limit=3,                           # number of returned entities
        output_fields=["text"]
    )

    for result in res:
        example = (result[0]["entity"]["text"])
    print("retrieved")
    return content, example
    
# you might want a reranker in here to discard poor retrieval



def call_model(example, content):
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    hugging_face_token = "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"
    
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device="cpu",
        #device_map="auto",
        token = hugging_face_token
    )
    
    messages = [
        {"role": "system", "content": example}, #"You are a pirate chatbot who always responds in pirate speak!"},
        {"role": "user", "content": content},
    ]
    
    prompt = pipeline.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
    )
    
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
        prompt,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    print("generation complete")
    return (outputs[0]["generated_text"][len(prompt):])

def call_RAG(content):
    hugging_face_token = "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"
    device="cpu"
    # init model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('maidalun1020/bce-embedding-base_v1', token=hugging_face_token)
    model = AutoModel.from_pretrained('maidalun1020/bce-embedding-base_v1', token=hugging_face_token)
    model.to(device)
    example = query_vect_db(content, tokenizer, model)
    return call_model(example, content)

def main():
    st.title("Pytorch 2.2 Docs RAG app")
    #web_input = st.text_input("Enter a website link")
    user_question = st.text_input("Ask a Pytorch question")
    st.button("Submit")
    ans = call_RAG(user_question)
    st.write(ans)
    
if __name__ == "__main__":
    main()

