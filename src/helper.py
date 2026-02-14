from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
from langchain_core.documents import Document


#Extract Data from the PDF File
def load_pdf_file(data):
    loader = DirectoryLoader(
        data,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    documents =loader.load() 
    return documents

#reduce the size of the documents by keeping only the page content and source metadata
def filter_to_minimal_docs(docs):
    #given a list of documents, return a new list of documents with only the page content and source metadata
    minimal_docs = [] #ek empty list banaya hai jisme minimal docs store honge
    for doc in docs:
        src = doc.metadata.get("source") #source ko metadata se nikalna hai, baki hata dena hai
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src} #meta data ma sirf source rakhna hai, baki hata dena hai
            ) #Document class ka object banaya hai jisme page_content aur metadata hai
        )
    return minimal_docs


#Splitting the documents into smaller chunks using RecursiveCharacterTextSplitter
def text_split(minimal_docs):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100   
    )
    text_chunks=text_splitter.split_documents(minimal_docs)
    return text_chunks
    
# this function will download the hugging face model and return the embedding 
# object which we can use to create vector store and generate embeddings for 
# our text chunks
def download_embeddings():
    model_name ="sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings

#copy all the above functions in helper.py and import them in store_index.py to use them there. This will help us to keep our code organized and modular. We can also reuse these functions in other parts of our code if needed.