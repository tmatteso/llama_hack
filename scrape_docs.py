import os
import requests
from bs4 import BeautifulSoup
import bs4
# URL of the PyTorch documentation
base_url = 'https://pytorch.org/docs/stable/'

print(help(BeautifulSoup))
# print("requests version:", requests.__version__)
# print("BeautifulSoup version:", bs4.__version__)
raise Error

# Fetch the content of the base URL
response = requests.get(base_url)
soup = BeautifulSoup(response.text, 'html.parser')

# Find all links in the navigation sidebar
links = soup.select('.toctree-l1 a')

# Iterate over each link
for link in links:
    # Get the URL of the documentation page
    doc_url = base_url + link['href']
    print(doc_url)
    #doc_url = "https://pytorch.org/docs/stable/notes/mps.html"
    doc_url = "https://pytorch.org/docs/stable/torch_cuda_memory.html"
    # Fetch the content of the documentation page
    response = requests.get(doc_url)
    doc_soup = BeautifulSoup(response.text, 'html.parser')

    # Get the main content of the page
    #content_element = doc_soup.select_one('.body')
    content_element = doc_soup.select_one('.main-content')

    if content_element is not None:
        content = content_element.get_text()
        print(content)
        raise Error
        # Create a text file and write the content to it
        with open(f'docs/{link.text}.txt', 'w') as f:
            f.write(content)
        print(1)
    else:
        content = ''
        print(2)
    

