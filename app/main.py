from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
import os, json, gc
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv

# --- Load environment ---
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=dotenv_path)
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
if langchain_api_key:
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["LANGCHAIN_TRACING_V2"] = "true"

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

app = FastAPI(title="ABAP Code Explanation API")


# ---- Strict input models ----
class ABAPSnippet(BaseModel):
    pgm_name: Optional[str] = None
    inc_name: Optional[str] = None
    type: Optional[str] = None
    name: Optional[str] = None
    code: str

    @field_validator("code", mode="before")
    @classmethod
    def clean_code(cls, v):
        return v.strip() if v else v


# ---- Summarizer ----
def summarize_snippet(snippet: ABAPSnippet) -> dict:
    return {
        "pgm_name": snippet.pgm_name,
        "inc_name": snippet.inc_name,
        "unit_type": snippet.type,
        "unit_name": snippet.name,
        "code": snippet.code,
    }


# --- Load RAG knowledge base ---
rag_file_path = os.path.join(os.path.dirname(__file__), "rag_knowledge_base.txt")
loader = TextLoader(file_path=rag_file_path, encoding="utf-8")
documents = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=20000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

# Vectorstore
embedding = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(docs, embedding)
retriever = vectorstore.as_retriever()


def cleanup_memory(*args):
    for var in args:
        del var
    gc.collect()


def build_chain(snippet: ABAPSnippet):
    """Builds and returns the chain with retrieved context injected."""
    retrieved_docs = retriever.get_relevant_documents(snippet.code)
    retrieved_context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    SYSTEM_MSG = "You are a precise ABAP reviewer and explainer. Respond in strict JSON only."

    USER_TEMPLATE = """
You are an SAP ABAP Developer with 20 years of experience.
Make sure to **Have all fields of select query with all its conditions. And also all the different conditions in Code**
Based on the RAG context and ABAP code,generate a complete and professionally 
formatted explaination .

Return ONLY strict JSON:
{{
  "explanation": "<concise explanation of ABAP code>"
}}

Unit metadata:
- Program: {pgm_name}
- Include: {inc_name}
- Unit type: {type}
- Unit name: {name}

System context (from knowledge base):
{retrieved_context}

Snippet JSON:
{context_json}
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_MSG),
        ("user", USER_TEMPLATE),
    ])

    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
    parser = JsonOutputParser()
    return prompt | llm | parser, retrieved_context


def llm_explain(snippet: ABAPSnippet):
    ctx_json = json.dumps(summarize_snippet(snippet), ensure_ascii=False, indent=2)
    chain, retrieved_context = build_chain(snippet)
    return chain.invoke({
        "context_json": ctx_json,
        "pgm_name": snippet.pgm_name,
        "inc_name": snippet.inc_name,
        "type": snippet.type,
        "name": snippet.name,
        "retrieved_context": retrieved_context,
    })


@app.post("/explain-abap")
async def explain_abap(snippets: List[ABAPSnippet]):
    results = []
    for snippet in snippets:
        try:
            llm_result = llm_explain(snippet)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")

        results.append({
            "pgm_name": snippet.pgm_name,
            "inc_name": snippet.inc_name,
            "type": snippet.type,
            "name": snippet.name,
            "code": "",
            "purpose": llm_result.get("explanation", ""),
        })

    return results


@app.get("/health")
def health():
    return {"ok": True, "model": OPENAI_MODEL}
