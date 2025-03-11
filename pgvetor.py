
from langchain_postgres.vectorstores import PGVector
from needs import llm, embeddings ,connection, collection_name
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


docs_path = "./knowledge_base"
chunck_size = 1000
chunck_overlap = 200

loader= DirectoryLoader(docs_path, glob="**/*.*")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunck_size, chunk_overlap=chunck_overlap)
texts = text_splitter.split_documents(documents)


vector_store = PGVector.from_documents(
    embedding=embeddings,
    collection_name=collection_name,
    connection=connection,
    documents=texts,
    use_jsonb=True
)