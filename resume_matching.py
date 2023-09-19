import pandas as pd
import numpy as np
import PyPDF2
import os



# load the parsed resume data from the root folder
def extract_text(pdf_file_path):
    text = ""
    with open(pdf_file_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfFileReader(pdf_file)
        for page_num in range(pdf_reader.numPages):
            text += pdf_reader.getPage(page_num).extractText()
    return text    

def extract_data(root_folder):
    resume_data = []
    for root, _, files in os.walk(root_folder):
        for file_name in files:
            if file_name.endswith('.pdf'):
                file_path = os.path.join(root, file_name)
                resume_text = extract_text(file_path)
                resume_data.append({'File Name': file_name, 'Resume Content': resume_text})
    return resume_data

root = "PATH TO DATA FOLDER"

resume_data = extract_data(root)
df = pd.DataFrame(resume_data)

def clean(string):
    return ' '.join(i for i in string.split('\n'))

df['Resume Content'] = df['Resume Content'].apply(lambda x: clean(x))



# load the first 15 job description from huggingface dataset
dataset = pd.read_html('https://huggingface.co/datasets/jacob-hugging-face/job-descriptions/viewer/default/train?row=2')
job_descriptions = dataset[0]['job_description (string)'][:15]



# Load the Bert Tokenizer and tokenize all cv content and job description
from transformers import DistilBertTokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
token_cv_tokens = [tokenizer(text, padding=True, truncation=True, return_tensors="pt") for text in df['Resume Content'][:10]]
token_job_descriptions = [tokenizer(text, padding=True, truncation=True, return_tensors="pt") for text in job_descriptions]



# Load Bert model and create embedding from cv token and job descroption token
from transformers import DistilBertModel
model = DistilBertModel.from_pretrained("distilbert-base-uncased")
embeddings_job_descriptions = [model(**input_dict).last_hidden_state.mean(dim=1).detach().numpy() for input_dict in token_job_descriptions]
embeddings_cv_token = [model(**input_dict).last_hidden_state.mean(dim=1).detach().numpy() for input_dict in token_cv_tokens]



# Find cosine similarity between cv embeddings and job description embeddings
# Save the similarity score and map it with the file name
from sklearn.metrics.pairwise import cosine_similarity
final = []
name = {}
for i , embeddings_jd in enumerate(embeddings_job_descriptions):
    temp = []
    for j , cv_embeddings in enumerate(embeddings_cv_token):
        x = cosine_similarity(embeddings_jd, cv_embeddings)
        temp.append(x[0][0])
        name[(i , x[0][0])] = j
    temp.sort(reverse=True)
    temp2 = [df.iloc[name[(i , k)] , 0] for k in temp[:5]]
    final.append(temp2)
    
    
    
# Print top 5 match for every job description
for i , j in enumerate(final):
    print(f'Top 5 Match for job description {i+1} - {job_descriptions[i]}')
    print(final[i])
