import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

# Initialize conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

load_dotenv()

# UI Setup
st.set_page_config(
    page_title='BookGenie',
    page_icon='📚',
    layout='wide'
)
st.title("📖 BookGenie: Chat with your Library")
st.markdown("*Ask questions, get answers—straight from your books.*")

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Load knowledge base
embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Get book titles
unique_docs = {
    doc.metadata['title'].split('.pdf')[0].split('(')[0].strip()
    for doc in vectorstore.docstore._dict.values() 
    if 'title' in doc.metadata
}
books = "\n".join([f"- {title}" for title in unique_docs])

# LLM setup
llm = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="deepseek/deepseek-chat-v3-0324:free"
)

# Define prompt template with context support
system_prompt = f"""
You are BookGenie, trained on these books:
{books}

**Conversation History:**
{{history}}

**Context:**
{{context}}

**Current Question:**
{{input}}

Rules:
1. Maintain natural conversation flow
2. Cite sources like: "In [title]..."
3. Keep answers concise but helpful
"""

prompt = ChatPromptTemplate.from_template(system_prompt)

# Chat interface
if query := st.chat_input("Ask BookGenie about your books:"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(query)
    
    # Generate response
    QA_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, QA_chain)
    
    # Format conversation history
    history = "\n".join(
        f"{msg['role']}: {msg['content']}" 
        for msg in st.session_state.messages[-6:]  # Last 6 exchanges
    )
    
    response = rag_chain.invoke({
        "input": query,
        "history": history
    })
    
    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response["answer"])
    
    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})