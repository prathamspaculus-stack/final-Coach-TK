from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import uuid
import mysql.connector
import json
import os
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Annotated, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

load_dotenv()

os.environ["LANGCHAIN_PROJECT"] = "LANGGRAPH_API"

app = FastAPI(title="Coach TK")

THREAD_ID = str(uuid.uuid4())

parser = StrOutputParser()

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings,
    collection_name="podcast_chunks"
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)



# PROMPT
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



#DATABASE CONNECTION AND FUNCTION

mysql_conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="newpassword",
    database="coachtk"
)
mysql_cursor = mysql_conn.cursor()

def save_qa(thread_id, question, answer, chunk, chunk_timestamps):
    sql = """
    INSERT INTO coach_chat_logs
    (thread_id, user_question, ai_answer, chunk, chunk_timestamps)
    VALUES (%s, %s, %s, %s, %s)
    """
    mysql_cursor.execute(
        sql,
        (thread_id, question, answer, chunk, json.dumps(chunk_timestamps))
    )
    mysql_conn.commit()

def load_history(thread_id, limit=5):
    sql = """
    SELECT user_question, ai_answer
    FROM coach_chat_logs
    WHERE thread_id = %s
    ORDER BY created_at DESC
    LIMIT %s
    """
    mysql_cursor.execute(sql, (thread_id, limit))
    rows = mysql_cursor.fetchall()

    messages = []
    for q, a in rows:
        messages.append(HumanMessage(content=q))
        messages.append(AIMessage(content=a))

    return messages

def load_chat(thread_id):
    sql = """
    SELECT user_question, ai_answer
    FROM coach_chat_logs
    WHERE thread_id = %s
    ORDER BY created_at ASC
    """
    mysql_cursor.execute(sql, (thread_id,))
    rows = mysql_cursor.fetchall()

    messages = []
    for q, a in rows:
        messages.append({"role": "human", "content": q})
        messages.append({"role": "ai", "content": a})

    return messages



def extract_chunk_timestamps(docs, limit=5):
    timestamps = []
    for doc in docs:
        ts = doc.metadata.get("timestamp")
        if ts and ts not in timestamps:
            timestamps.append(ts)
        if len(timestamps) == limit:
            break
    return timestamps




# GRAPH LOGIC

class CoachAnswer(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    retrieved_docs: List[Document]
    context: str
    answer: str
    thread_id: str

def retrieve_docs(state: CoachAnswer):
    question = state["messages"][-1].content
    docs = retriever.invoke(question)
    return {"retrieved_docs": docs}

def context(state: CoachAnswer):
    chat_history = "\n".join(
        f"{m.type.upper()}: {m.content}" for m in state["messages"]
    )

    chunks = []
    for i, doc in enumerate(state["retrieved_docs"], 1):
        chunks.append(f"CONTENT {i}:\n{doc.page_content}")
        chunks.append(f"SOURCE LINK: {doc.metadata.get('reference_link', 'N/A')}")

    return {
        "context": f"""
CHAT HISTORY:
{chat_history}

CONTENT:
{"\n\n---\n\n".join(chunks)}
"""
    }

def answer(state: CoachAnswer):
    question = state["messages"][-1].content

    prompt = p1.format(
        content=state["context"],
        question=question
    )

    response = model.invoke(prompt)
    final_answer = parser.invoke(response)

    chunk_text = "\n\n---\n\n".join(
        doc.page_content for doc in state["retrieved_docs"]
    )

    chunk_timestamps = extract_chunk_timestamps(state["retrieved_docs"])

    save_qa(
        state["thread_id"],
        question,
        final_answer,
        chunk_text,
        chunk_timestamps
    )

    return {"answer": final_answer}

graph = StateGraph(CoachAnswer)
graph.add_node("retrieve_docs", retrieve_docs)
graph.add_node("context", context)
graph.add_node("answer", answer)

graph.add_edge(START, "retrieve_docs")
graph.add_edge("retrieve_docs", "context")
graph.add_edge("context", "answer")
graph.add_edge("answer", END)

workflow = graph.compile()


# API

class ChatRequest(BaseModel):
    message: str

class ContinueChatRequest(BaseModel):
    thread_id: Optional[str]=None
    message: str

class RetrieveRequest(BaseModel):
    question: str

# @app.get("/health")
# def health():
#     return {"status": "ok"}

@app.post("/rag/retrieve")
def rag_retrieve(req: RetrieveRequest):
    docs = retriever.invoke(req.question)

    results = []
    for doc in docs:
        results.append({
            "content": doc.page_content,
            "source_link": doc.metadata.get("reference_link"),
            "timestamp": doc.metadata.get("timestamp")
        })

    return {
        "question": req.question,
        "total_chunks": len(results),
        "chunks": results
    }

@app.post("/thread/create")
def create_thread():
    return {"thread_id": str(uuid.uuid4())}


@app.get("/chat/history/{thread_id}")
def chat_history(thread_id: str):
    messages = load_chat(thread_id)
    return {"history": messages}


@app.post("/chat/continue")
def continue_chat(req: ContinueChatRequest):
    past_messages = load_history(req.thread_id)

    state = {
        "messages": past_messages + [HumanMessage(content=req.message)],
        "thread_id": req.thread_id
    }

    result = workflow.invoke(
        state,
        config={"configurable": {"thread_id": req.thread_id}}
    )

    return {"answer": result["answer"]}

@app.post("/chat")
def chat(req: ChatRequest):

    past_messages = load_history(THREAD_ID)

    state = {
        "thread_id": THREAD_ID,
        "messages": past_messages + [HumanMessage(content=req.message)]
    }

    result = workflow.invoke(
        state,
        config={"configurable": {"thread_id": THREAD_ID}}
    )

    return {
        "thread_id": THREAD_ID,
        "answer": result["answer"]
        }


@app.post("/newchat")
def chat(req: ChatRequest):

    thread_id = str(uuid.uuid4())

    past_messages = load_history(thread_id)

    state = {
        "thread_id": thread_id,
        "messages": past_messages + [HumanMessage(content=req.message)]
    }

    result = workflow.invoke(
        state,
        config={"configurable": {"thread_id": thread_id}}
    )

    return {
        "thread_id": thread_id,
        "answer": result["answer"]
        }

@app.post("/chats")
def chats(req: ContinueChatRequest):

    thread_id = req.thread_id or str(uuid.uuid4())

    past_messages = load_history(thread_id)

    state = {
        "thread_id": thread_id,
        "messages": past_messages + [
            HumanMessage(content=req.message)
        ]
    }

    result = workflow.invoke(
        state,
        config={"configurable": {"thread_id": thread_id}}
    )

    return {
        "thread_id": thread_id,
        "answer": result["answer"]
    }

