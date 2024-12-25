import streamlit as st
import pandas as pd
from langchain.schema import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss

st.title("RAG Query Assistant")

# Upload CSV files
uploaded_files = st.file_uploader(
    "Upload one or more CSV files",
    type=["csv"],
    accept_multiple_files=True,
)

if uploaded_files:
    docs = []

    # Custom loader for CSV files
    for uploaded_file in uploaded_files:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(uploaded_file)
        # Convert rows into documents
        for _, row in df.iterrows():
            # Each row is converted into a Document
            doc_content = "\n".join([f"{col}: {val}" for col, val in row.items()])
            docs.append(Document(page_content=doc_content))

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
    user_query = st.text_input("Enter your query:")

    if user_query:
        # Retrieve relevant documents
        ret_doc = vector_store.similarity_search(user_query, k=3)

        if ret_doc:
            context = "\n".join([doc.page_content for doc in ret_doc])
            final_prompt = prompt.format_prompt(question=user_query, context=context).to_string()
            response = model.invoke(final_prompt).content

            # Display the response
            st.subheader("Answer:")
            st.write(response)

            # Display the context
            st.subheader("Retrieved Context:")
            for i, doc in enumerate(ret_doc):
                st.write(f"**Document {i + 1}:** {doc.page_content}")

        else:
            st.warning("No relevant context found in the documents.")
