import os
import requests
from bs4 import BeautifulSoup
import bs4
from transformers import AutoModel, AutoTokenizer
from pymilvus import MilvusClient, DataType
import torch

def chunk_text(text, max_length, tokenizer):
    # Tokenize the text
    #tokens = tokenizer.tokenize(text,return_tensors="pt")
    tokens = tokenizer.encode_plus(text,return_tensors="pt")
    # Split the tokens into chunks of max_length
    chunks = [tokens.input_ids[:, i:i+max_length//2] for i in range(0, (tokens.input_ids.shape[1]), max_length//2)]
    return chunks
    
def retrieve_docs(client, device = "cpu"):
    hugging_face_token = "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"
    # init model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('maidalun1020/bce-embedding-base_v1', token=hugging_face_token)
    model = AutoModel.from_pretrained('maidalun1020/bce-embedding-base_v1', token=hugging_face_token)
    model.to(device)
    
    # URL of the PyTorch documentation
    base_url = 'https://pytorch.org/docs/2.2/' # this can be changed
    # may be a good idea to get all the blog posts too
    #https://pytorch.org/blog/

    # Fetch the content of the base URL
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all links in the navigation sidebar
    links = soup.select('.toctree-l1 a')

    # eliminate all the links with "#"
    links = [s for s in links if "#" not in s['href']]
    print(len(links))
    # Iterate over each link
    for i in range(len(links)):
        link = links[i]
        doc_embeds, doc_chunks = [], []
        # Get the URL of the documentation page
        doc_url = base_url + link['href']
        print(doc_url, i/len(links))
        # Fetch the content of the documentation page
        response = requests.get(doc_url)
        doc_soup = BeautifulSoup(response.text, 'html.parser')
    
        # Get the main content of the page
        #content_element = doc_soup.select_one('.body')
        content_element = doc_soup.select_one('.main-content')
    
        if content_element is not None:
            content = content_element.get_text()
            chunks = chunk_text(content, 512, tokenizer)
            inputs_on_device = dict()
            for chunk in chunks:
                embeddings = model(chunk, return_dict=True).last_hidden_state[:, 0]
                embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)  # normalize
                doc_embeds.append(embeddings.half())
            doc_chunks += (chunks)
            
            fill_vect_db(doc_chunks, doc_embeds, client, tokenizer)



def setup_db():
    # 1. Set up a Milvus client
    client = MilvusClient(
        uri="http://localhost:19530"
    )  
    return client

def fill_vect_db(all_chunks, all_embeddings, client, tokenizer):
 
    # 2. Create a collection in quick setup mode
    client.create_collection(
        collection_name="llama2",
        dimension=768
    )
    # 4. Insert data into the collection
    # 4.1. Prepare data
    data = [{"id": i, "text": tokenizer.decode(all_chunks[i][0]),
             "vector": all_embeddings[i].flatten()} for i in range(len(all_embeddings))]

    res = client.insert(
        collection_name="llama2",
        data=data
    )
# need to start milvus before you try to set up the connection
# bash standalone_embed.sh start
def main():
    client = setup_db()
    retrieve_docs(client)
    print("all docs embedded")


if __name__ == "__main__":
    main()
