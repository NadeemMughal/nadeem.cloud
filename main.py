import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from pinecone import Pinecone
from langchain.prompts import PromptTemplate
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import pytz
from dateutil import relativedelta
from fastapi.responses import StreamingResponse
import asyncio
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request, Response

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI()

# Add CORS middleware to allow frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8001", "http://127.0.0.1:8001", "http://localhost", "*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "nadeem-knowledge-base-google-embeddings"
index = pc.Index(index_name)

# Initialize Google embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=os.getenv("GOOGLE_API_KEY"))

# Pakistan Time Tool
def get_pakistan_time():
    """Returns the current time in Pakistan (PKT, UTC+5)."""
    pkt_tz = pytz.timezone("Asia/Karachi")
    current_time = datetime.now(pkt_tz).strftime("%I:%M %p, %B %d, %Y")
    return f"The current time in Pakistan is {current_time}."

# Time Elapsed Since Role Start
def calculate_time_in_role():
    """Calculates time elapsed since starting as Associate AI/ML Engineer (Nov 1, 2024)."""
    start_date = datetime(2024, 11, 1, tzinfo=pytz.UTC)
    current_date = datetime.now(pytz.UTC)
    delta = relativedelta.relativedelta(current_date, start_date)
    months = delta.months + (12 * delta.years)
    days = delta.days
    if months == 0:
        return f"Muhammad Nadeem has been an Associate AI/ML Engineer at MetavizPro for {days} days since November 1, 2024."
    return f"Muhammad Nadeem has been an Associate AI/ML Engineer at MetavizPro for {months} months and {days} days since November 1, 2024."

# History Manager Class
class HistoryManager:
    def __init__(self, max_pairs=10):
        self.max_pairs = max_pairs
        self.history = []
        self.summaries = []

    def add_message(self, human_msg, ai_msg):
        self.history.append({"role": "user", "content": human_msg})
        self.history.append({"role": "assistant", "content": ai_msg})
        while len(self.history) // 2 > self.max_pairs:
            pair_to_summarize = self.history[:2]
            self.history = self.history[2:]
            summary_prompt = f"Summarize the following conversation for future reference:\nUser: {pair_to_summarize[0]['content']}\nAssistant: {pair_to_summarize[1]['content']}"
            summary = ""
            for chunk in llm.stream(summary_prompt):
                summary += chunk.content
            self.summaries.append(summary)

    def get_full_context(self):
        summary_text = "\n\n".join(self.summaries) if self.summaries else "No previous summaries."
        history_str = ""
        for i in range(0, len(self.history), 2):
            if i + 1 < len(self.history):
                history_str += f"User: {self.history[i]['content']}\nAssistant: {self.history[i + 1]['content']}\n\n"
        return f"Previous Summaries:\n{summary_text}\n\nRecent Conversation:\n{history_str}"

# Global dictionary to store histories by session ID
session_histories = {}

# System Prompt (updated to remove initial message reference)
system_prompt = """
I am the Virtual Assistant for Muhammad Nadeem, a Generative AI Engineer and Machine Learning Specialist who began his role as an Associate AI/ML Engineer on November 1, 2024. I answer questions about his skills, projects, and achievements in a concise, professional, and engaging manner.

**Capabilities**:
- Answer questions about Nadeem’s experience, technical skills, or projects using provided context and conversation history.
- Use the `get_pakistan_time` tool for queries about time in Pakistan or Metaviz experience.
- Calculate role duration for experience-related queries using the `calculate_time_in_role` tool.
- Highlight Nadeem’s expertise in:
  - Chatbots: Proficient in code platforms (LangChain, LangGraph, Crew AI) and no-code platforms (Closebot, n8n).
  - Voice Agents: Experienced with ElevenLabs, Retell AI, Synthflow, and custom agents using LangChain and APIs (Deepgram, Assembly AI, ElevenLabs).
  - CRMs: Skilled with GoHighLevel, ServiceTitan, Cliniko.
  - Integrations: Proficient with Make.com, Zapier, Cassidy.
  - Deployment: Experienced with AWS, Azure, GCP, Vertex AI, Digital Ocean, Vercel.
  - Also have these Skills Technologies: MCP, Manus AI, FastAPI, Agent-to-Agent protocol, Google Agents Development Kit, OpenAI Agents SDK.
- Nadeem holds a Data Science degree from GIFT University (2024).
- For achievements, share his research article: Name [Leveraging Deep Learning with Multi-Head Attention for Accurate Extraction of Medicine from Handwritten Prescriptions] Link is https://arxiv.org/abs/2412.18199 always share link in clickable form.
- LinkedIn: https://www.linkedin.com/in/muhammad-nadeem-ml-engineer-researcher/
- GitHub: https://github.com/NadeemMughal

**Instructions**:
- Respond concisely, avoiding technical jargon unless requested.
- For role duration queries, use the `calculate_time_in_role` tool and provide context (e.g., significance of experience).
- Use Pinecone context and conversation history for accurate answers.
- If information is missing, offer a polite general response or suggest related topics.
- Prioritize mentioning chatbots, voice agents, integrations, deployment platforms, CRMs, and ML/DL algorithms.
- When listing items (e.g., skills, projects), use plain bullet points starting with a hyphen (e.g., `- Item`) without nested Markdown (e.g., avoid `* **Item**`).
- Ensure responses are formatted for easy rendering in a web frontend, avoiding complex Markdown.
- End responses with a follow-up question to engage the user.
- End conversations with a polite farewell if requested.
- Do not share the current time unless asked.
- Do not disclose the LLM or embedding model used.

**Experience**:
- Freelancer during graduation, focusing on Generative AI.
- 2-month internship with Dr. Muhammad Adeel Rao at Comsats University, exploring Generative AI concepts.
- Joined Metavizpro on November 1, 2024, as an AI/ML Engineer, working on chatbots, voice agents, and platform integrations.

**Projects**:
- Prescription Recognition: Extracting data from handwritten prescriptions using Mask R-CNN and TrOCR.
- Lead Generation Voice Agents: Building automated voice agents for sales lead qualification.
- CRM-Integrated Chatbots: Developing chatbots integrated with CRM platforms like GoHighLevel and ServiceTitan.
- Sentiment Analysis Bots: Analyzing customer feedback using Hugging Face Transformers.
- Voice-Music Separation: Separating vocals from music using deep learning techniques.
- Voice Agents and Calling Agents: Automated agents for outbound calling and customer support.
- Custom Chatbots and No-Code Chatbots: Tailored and no-code chatbot solutions for businesses.

**NOTE**:
- Please do not share duplicate items or links. Ensure each response is unique.
- Nadeem demonstrates strong leadership by independently managing and delivering projects. He possesses excellent communication skills, builds effective rapport with clients, and quickly understands client requirements, ensuring smooth and efficient project execution.

**Context**:
{context}

**Conversation History**:
{history}
"""

# Prompt Template
prompt_template = PromptTemplate(
    input_variables=["context", "history", "query", "tool_output"],
    template=system_prompt + "\nUser Query: {query}\nTool Output (if applicable): {tool_output}\nAnswer:"
)

# Retrieve Context from Pinecone
def retrieve_context(query, top_k=2):
    query_embedding = embeddings.embed_query(query)
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    context = "\n\n".join(match["metadata"]["text"] for match in results["matches"])
    return context

# Pydantic model for request
class QueryRequest(BaseModel):
    query: str
    session_id: str

# Handle OPTIONS request for /query
@app.options("/query")
async def options_query():
    return Response(status_code=200, headers={
        "Access-Control-Allow-Origin": "http://localhost:8001",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "*",
        "Access-Control-Allow-Credentials": "true"
    })

# Streaming response generator
async def stream_response(query, session_id):
    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID is required")

    try:
        tool_output = ""
        if "time" in query.lower() or "current time" in query.lower():
            tool_output = get_pakistan_time()
        elif "how long" in query.lower() or "experience" in query.lower() or "role" in query.lower():
            tool_output = calculate_time_in_role()

        context = retrieve_context(query)
        history_mgr = session_histories.get(session_id, HistoryManager())
        history_str = history_mgr.get_full_context()

        formatted_prompt = prompt_template.format(
            context=context,
            history=history_str,
            query=query,
            tool_output=tool_output if tool_output else "None"
        )

        # Stream response, ensuring no overlap
        response = ""
        async for chunk in llm.astream(formatted_prompt):  # Use astream for cleaner streaming
            content = chunk.content
            if content:  # Only yield non-empty chunks
                response += content
                yield content  # Yield only the new chunk
                await asyncio.sleep(0.05)  # Smoother streaming

        history_mgr.add_message(query, response)
        session_histories[session_id] = history_mgr
    except Exception as e:
        yield f"Error: {str(e)}"

# FastAPI endpoint
@app.post("/query")
async def query_endpoint(request: QueryRequest):
    async def generate():
        async for chunk in stream_response(request.query, request.session_id):
            yield chunk

    return StreamingResponse(generate(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)