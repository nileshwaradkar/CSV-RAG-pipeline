from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss


# List of PDF URLs to load
csv_files = [
    "rag_pipeline_data.csv"
    ]

docs = []
for file_path in csv_files:
    loader = CSVLoader(file_path)
    docs.extend(loader.load())

# Split documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
all_splits = text_splitter.split_documents(docs)

# Initialize embeddings and vector store
embeddings = OllamaEmbeddings(model="phi3:latest")
index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)

# Add documents to the vector store
vector_store.add_documents(documents=all_splits)

# Initialize the LLM model
model = ChatOllama(model="phi3:latest")

# Create a prompt template
prompt = ChatPromptTemplate.from_template(
    """
    You are a knowledgeable assistant specializing in question-answering. 
    Please utilize the provided context to formulate your response. 
    If the answer is not available, simply state that you do not know.
    Limit your response to three concise sentences.

    Question: {question}
    Context: {context}
    Answer:
    """
)

# Query input
question = input("Enter your query: ")

# Retrieve relevant documents
ret_doc = vector_store.similarity_search(question, k=3)

#for doc in ret_doc:
#    print("Content: ",doc.page_content)
#    print("Metadata" ,doc.metadata)
    
# Display results
if ret_doc:
    context = "\n".join([doc.page_content for doc in ret_doc])
    final_prompt = prompt.format_prompt(question=question, context=context).to_string()
    response = model.invoke(final_prompt).content
    print("Answer:", response)
else:
    print("No relevant context found in the documents.")    
