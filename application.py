import streamlit as st

#from datasets import load_dataset
#ds = load_dataset("ccdv/pubmed-summarization", "document")

#preprocessing
#cleaning
import re
def clean(txt):
    txt = re.sub(r'[^a-zA-Z0-9\s]','',txt)
    return txt
#removing stopwords
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
def remStopWords(txt):
    stoplist = set(stopwords.words('english'))
    word_tokens = word_tokenize(txt)
    filtered_text = [word for word in word_tokens if word.lower() not in stoplist]
    filtered_text = ' '.join(filtered_text)
    return filtered_text
#lemmatization
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')
# Function to map POS tag to first character lemmatize() accepts
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts."""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {
        'J': wordnet.ADJ,
        'N': wordnet.NOUN,
        'V': wordnet.VERB,
        'R': wordnet.ADV
    }
    return tag_dict.get(tag, wordnet.NOUN)

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    word_tokens = word_tokenize(text)
    lemmatized_text = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in word_tokens]
    return ' '.join(lemmatized_text)
#using langchain to import openai for summary
from langchain_community.llms import OpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
def generate_response(txt,api_key):
    # Instantiate the LLM model
    llm = OpenAI(temperature=0, openai_api_key=api_key)

    # Split text
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(txt)

    # Create multiple documents
    docs = [Document(page_content=t) for t in texts]

    # Text summarization
    chain = load_summarize_chain(llm, chain_type='map_reduce')
    return chain.run(docs)
def generate_response2(txt,api_key):
    # Instantiate the LLM model
    llm = OpenAI(temperature=0.5, openai_api_key=api_key)

    # Split text
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(txt)

    # Create multiple documents
    docs = [Document(page_content=t) for t in texts]

    # Text summarization
    chain = load_summarize_chain(llm, chain_type='stuff')
    return chain.run(docs)
#streamlit
from PIL import Image
import requests
from io import BytesIO
st.set_page_config(page_title=" PubMed Article Summarization")
st.title("PubMed Article Summarization")
banner_url = 'https://www.biava.me/wp-content/uploads/2015/12/PubMed-Logo.png'
response1 = requests.get(banner_url)
image = Image.open(BytesIO(response1.content))
st.image(image, caption='OpenAI Integrated Summarization')

text = st.text_area("Enter Your Article",'',height=200)
text = clean(text)
text = remStopWords(text)
text = lemmatize_text(text)
summarized = []
option = st.radio(
    label = "Type of Summary: " ,
    options = ("Brief","Detailed")
)
option2 = st.radio(
    label  = "Generate Dall-E Image: " ,
    options = ("Yes","No")
)
#image generation using openai dall-e
# Function to generate image from text using OpenAI's DALL-E
import openai
def generate_image(prompt):
    openai.api_key = api_key
    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size="1024x1024"
    )
    image_url = response['data'][0]['url']
    return image_url
with st.form('summarize_form', clear_on_submit=True):
    api_key = st.text_input('OpenAI API Key', type='password', disabled=not text)
    submitted = st.form_submit_button('Submit')
    if submitted and api_key.startswith('sk-'):
        with st.spinner('Calculating...'):
            if option == "Brief":
                response = generate_response(text,api_key)
            elif option == "Detailed":
                response = generate_response2(text,api_key)
        summarized.append(response)
        if len(summarized):
            st.info(response)
        if(option2 == "Yes"):
            image_url = generate_image(response)
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
            st.image(image, caption=response)
        del api_key

