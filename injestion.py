from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def injest_docs():
    loader = ReadTheDocsLoader("./langchain-docs/api.python.langchain.com/en/latest")

    raw_documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=50
    )

    documents = text_splitter.split_documents(raw_documents)

    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    PineconeVectorStore.from_documents(
        documents, embeddings, index_name="document-helper"
    )

    print(f"Loading {len(documents)} into vectorstore")

if __name__ == '__main__':
    injest_docs()
