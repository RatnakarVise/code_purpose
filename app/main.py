from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
import os, json
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
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
    # context: Optional[str] = None
    # suggested_improvements: Optional[List[str]] = Field(default_factory=list)

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
        # "context": snippet.context,
        # "suggested_improvements": snippet.suggested_improvements,
    }


# ---- LangChain Prompt ----
SYSTEM_MSG = "You are a precise ABAP reviewer and explainer. Respond in strict JSON only."

USER_TEMPLATE = """
You are evaluating an ABAP code snippet.

Your tasks:
1) Provide a concise **explanation**:
   - Purpose: what does this ABAP code do?
   - Key statements: SELECT, READ TABLE, LOOP, SORT, FORM/METHOD usage.

2) Provide an actionable **LLM remediation prompt**:
   - Include original code snippet.
   - Create the detailed explanation using the context provided.

SCOPE & CHECKLIST (consider each item; mention only those that apply)
- Database access: SELECT/SELECT SINGLE, field lists vs SELECT *, key usage, WHERE clauses, ORDER BY, buffering, FOR ALL ENTRIES pre-check, joins vs nested selects, SELECT in LOOP (N+1), UP TO 1 ROWS semantics.
- Internal tables: types (standard/sorted/hashed), keys, SORT vs READ TABLE, table expressions itab[ key ], line_exists( ), TRANSPORTING NO FIELDS, binary search correctness, memory/perf in loops.
- Control flow & modularization: LOOP/ENDLOOP, nested loops, FORM/METHOD/CLASS usage, interface design, reusability.
- Error handling: sy-subrc patterns, exception classes, MESSAGE vs raising exceptions, ASSERT/CHECK preconditions.
- Modernization (Clean ABAP): inline DATA, VALUE/NEW, CORRESPONDING #, string templates, FILTER/REDUCE, avoiding obsolete statements.
- S/4HANA simplifications: deprecated objects/APIs, table/model changes, CDS/AMDP suitability (only if relevant).
- Performance: algorithmic complexity, sorting once vs repeated sorts, use proper table types/keys, avoid unnecessary conversions.
- Security & compliance: AUTHORITY-CHECK, input validation, escaping, RFC/background safety, COMMIT/ROLLBACK usage (do not relocate).
- Internationalization/Unicode/time: dates/times/time zones/numeric precision, string handling.
- Testability: ABAP Unit seams, dependency inversion, test doubles.


RULES
- Be factual and code-grounded. If any input is missing, proceed using only what is present and state that limitation.
- Do NOT introduce new business logic; keep behavior equivalent 

Return ONLY strict JSON:
{{
  "explanation": "<concise explanation of ABAP code>",
}}

Unit metadata:
- Program: {pgm_name}
- Include: {inc_name}
- Unit type: {type}
- Unit name: {name}

System context:
{context_json}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_MSG),
    ("user", USER_TEMPLATE),
])

llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
parser = JsonOutputParser()
chain = prompt | llm | parser


def llm_explain(snippet: ABAPSnippet):
    ctx_json = json.dumps(summarize_snippet(snippet), ensure_ascii=False, indent=2)
    return chain.invoke({
        "context_json": ctx_json,
        "pgm_name": snippet.pgm_name,
        "inc_name": snippet.inc_name,
        "type": snippet.type,
        "name": snippet.name,
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
            "code": snippet.code,  # keep raw ABAP code outside response for safety
            "purpose": llm_result.get("explanation", ""),
            # "llm_prompt": llm_result.get("llm_prompt", "")
        })

    return results


@app.get("/health")
def health():
    return {"ok": True, "model": OPENAI_MODEL}
