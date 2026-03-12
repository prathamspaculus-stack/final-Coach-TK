from dotenv import load_dotenv
import uuid
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, List
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, RemoveMessage
import os
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_core.documents import Document

os.environ["LANGCHAIN_PROJECT"] = 'LANGGRAPH PRA'

load_dotenv()

parser = StrOutputParser()

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
    )

embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)
        
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings,
    collection_name="podcast_chunks"
)

retriever = vectorstore.as_retriever(
    search_type = "similarity",
    search_kwargs={"k": 5}
)

p1 = PromptTemplate(
    input_variables=["question", "content"],
    template="""
You are a practical coach Terry kim, not a teacher.

RULES:
- Use the provided content as PRIMARY source
- You may also use information shared earlier by the user in this conversation
- Do NOT add outside knowledge
- Do NOT sound academic or robotic
- Keep language simple and easy to understand
- Be clear and practical

LINK RULE:
- Suggest the SOURCE LINK only if it helps the user learn more
- Mention multitple link
- Do not force a link if not needed

COACHING STYLE:
- Explain like you are guiding a beginner
- Use one small, real-life example if helpful
- Keep the answer focused

CONTEXT (use only this):
{content}

USER QUESTION:
{question}

ANSWER:
"""
)

class CoachAnswer(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    retrieved_docs: List[Document]
    context: str
    answer: str

def retrieve_docs(state: CoachAnswer):
    question = state["messages"][-1].content
    docs = retriever.invoke(question)
    return {"retrieved_docs": docs}

def context(state: CoachAnswer):
    
    chunks = []
    for i, doc in enumerate(state['retrieved_docs'],1):
        chunks.append(f"CONTENT {i}:\n{doc.page_content}")
        chunks.append(f"SOURCE LINK: {doc.metadata.get('reference_link', 'N/A')}")

    return {
        "context": f"""
        CONTENT:
        {"\n\n---\n\n".join(chunks)}
"""
    }

def answer(state: CoachAnswer):
    question = state["messages"][-1].content

    prompt = p1.format(
        content = state["context"],
        question=question
    )

    response = model.invoke(prompt)
    final_answer = parser.invoke(response)

    return {'answer': final_answer}

graph = StateGraph(CoachAnswer)
graph.add_node("retrieve_docs", retrieve_docs)
graph.add_node("context", context)
graph.add_node("answer", answer)

graph.add_edge(START, "retrieve_docs")
graph.add_edge("retrieve_docs", "context")
graph.add_edge("context", "answer")
graph.add_edge("answer", END)

workflow = graph.compile()

while True:
    user_message = input("You: ")

    if user_message.lower() in ["exit", "quit", "bye"]:
        print("Session ended.")
        break

    result = workflow.invoke(
        {
            "messages": [
                HumanMessage(content=user_message)
            ]
        }
    )

    print("TK:", result["answer"])
