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
- 12 real projects built and published on GitHub:

  1. AI Career Coach Agent — multi-agent system (6 agents), Claude Sonnet 4.6,
     RAG pipeline, ChromaDB, parallel search, Gradio UI
  2. StayEasy RAG — hotel support chatbot, RAG, vector embeddings, OpenAI
  3. Text-to-SQL Fine-tuning — GPT-4o-mini fine-tuned for natural language to SQL
  4. DeepSeek SQL Fine-tuning — DeepSeek Coder 1.3B + LoRA for SQL generation
  5. Groq YouTube Summarizer — fast video summaries using Groq API
  6. Story Generator — GPT-4o-mini + DALL-E 3 + TTS multi-modal app
  7. Brochure Generator — website scraper + PDF generator with Gradio
  8. Ollama Webpage Summarizer — 100% local summarizer using Llama 3.2
  9. Chatbot Conversation — multi-model dialogue GPT-4 vs Ollama
  10. AI Tutor — AI concepts explainer using Ollama
  11. Student Data Generator — synthetic data with OpenAI + HuggingFace
  12. BLIP Image Captioning — automatic image captions using Salesforce BLIP

- LLM APIs: Anthropic Claude, OpenAI, Groq, Ollama
- AI Tools: Cursor, Claude, ChatGPT, GitHub Copilot
- Dev tools: Python, LangChain, HuggingFace, ChromaDB, Gradio, Modal, DALL-E, TTS
- Key strength: Ships AI products fast using LLM APIs and AI-assisted tools
- Location: Open to remote, hybrid, and onsite — will negotiate after offer
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
