import anthropic
import json
from dotenv import load_dotenv
from rag.vector_store import VectorStore

load_dotenv()

client = anthropic.Anthropic()

# This tells Claude what you already know
# So it won't say "learn Python" when you already know Python
CANDIDATE_CONTEXT = """
Candidate background:
- NOT a traditional software engineer — builds using AI tools (Claude, Cursor, ChatGPT)
- Completed Ed Donner's 8-week LLM Engineering course
- Built 12+ real projects: RAG pipelines, fine-tuned GPT-4o-mini and DeepSeek 1.3B (LoRA),
  multi-agent systems, vector stores (ChromaDB), Gradio UIs, Modal deployment,
  YouTube summarizer, SQL fine-tuning, hotel support chatbot, story generator
- LLM APIs: Anthropic Claude, OpenAI, Groq, Ollama
- AI Tools: Cursor, Claude, ChatGPT, GitHub Copilot — builds with AI, not around it
- Dev tools: Python, LangChain, HuggingFace, ChromaDB, Gradio, Modal
- Looking for: roles at AI-first startups where AI tools are used daily
- Location: Remote, willing to travel to California
- Key strength: Can build and ship AI-powered products quickly using LLM APIs and AI tools
"""


class SkillGapAgent:

    def __init__(self):
        # Connect to the resume vector store we built in rag/
        self.store = VectorStore(collection="resume")

    def analyze(self, job: dict, role: str = "AI Engineer", position_type: str = "Junior AI Engineer") -> list[str]:
        """Compare resume to job and return list of missing skills."""

        # Step 1: Use RAG to find the most relevant resume chunks
        # We search using the job description as the query
        resume_chunks = self.store.query(job["description"], n=5)
        resume_context = "\n---\n".join(resume_chunks)

        # Step 2: Ask Claude to find the gaps
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=f"""You are a career coach helping a non-traditional candidate
land a {position_type} role as a {role} at an AI-first startup.
This candidate builds using AI tools (Claude, Cursor, ChatGPT) — NOT traditional coding.
Find REAL skill gaps specific to this job — practical things they are missing.
Do NOT suggest: "learn Python", "learn to code", or anything they clearly already know
(LLMs, RAG, fine-tuning, ChromaDB, Gradio, Claude API, OpenAI API).
Focus on gaps that can be closed with short online courses or practice.
Return ONLY a JSON array of strings, no extra text.""",
            messages=[{
                "role": "user",
                "content": f"""
Candidate Background:
{CANDIDATE_CONTEXT}

Relevant Resume Sections:
{resume_context}

Job Requirements:
{job['description']}

What are the top 5 skills missing for this '{position_type} - {role}' role?
Return as JSON array: ["skill1", "skill2", "skill3", "skill4", "skill5"]
"""
            }]
        )

        # Step 3: Parse the JSON array from Claude's response
        text  = response.content[0].text.strip()
        start = text.find("[")
        end   = text.rfind("]") + 1

        # Safety check — return empty list if parsing fails
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            print(f"Could not parse skill gaps: {text}")
            return []
