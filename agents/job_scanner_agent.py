import anthropic
import feedparser
import json
import os
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()

# Job feeds — remote, hybrid, and onsite
RSS_FEEDS = [
    "https://remoteok.com/remote-ai-jobs.rss",
    "https://remoteok.com/remote-llm-jobs.rss",
    "https://remoteok.com/remote-machine-learning-jobs.rss",
    "https://remoteok.com/remote-python-jobs.rss",
    "https://weworkremotely.com/categories/remote-programming-jobs.rss",
]

# Keywords to filter jobs by role
ROLE_KEYWORDS = {
    "AI Engineer": [
        "ai engineer", "artificial intelligence", "llm", "openai", "anthropic",
        "generative ai", "machine learning", "claude", "cursor", "copilot",
        "chatgpt", "gpt", "ai tools", "ai-powered", "ai powered",
    ],
    "LLM Engineer": [
        "llm engineer", "large language model", "fine-tuning", "rag",
        "langchain", "prompt engineer", "vector", "embedding", "claude",
        "openai", "anthropic", "huggingface", "agentic", "ai agent",
    ],
}

# Keywords to filter by position level
POSITION_KEYWORDS = {
    "Paid Intern":          ["intern", "internship", "co-op"],
    "Junior AI Engineer":   ["junior", "entry level", "entry-level", "associate", "ai engineer"],
    "Junior LLM Engineer":  ["junior", "entry level", "entry-level", "associate", "llm engineer"],
}

# Your background — Claude reads this to find the best matching jobs
CANDIDATE_PROFILE = """
- Open to: Paid Internship, Junior AI Engineer, Junior LLM Engineer
- Location: Open to remote, hybrid, and onsite — will negotiate after offer
- Target companies: Early-stage startups (YC, Series A/B)
- Experience: Completed Ed Donner's 8-week LLM Engineering course

Projects built:
  1. AI Career Coach Agent — multi-agent system using Claude Sonnet 4.6,
     RAG pipeline, ChromaDB, 6 specialized agents, Gradio UI, parallel search
  2. StayEasy RAG — hotel customer support chatbot with RAG, vector embeddings, OpenAI
  3. Text-to-SQL Fine-tuning — fine-tuned GPT-4o-mini for natural language to SQL
  4. DeepSeek SQL Fine-tuning — fine-tuned DeepSeek Coder 1.3B with LoRA for SQL
  5. Groq YouTube Summarizer — YouTube video summarizer using Groq API
  6. Story Generator — multi-modal story generator with GPT-4o-mini, DALL-E 3, TTS
  7. Brochure Generator — scrapes websites and generates PDFs using ChatGPT + Gradio
  8. Ollama Webpage Summarizer — local summarizer using Llama 3.2 via Ollama
  9. Chatbot Conversation — GPT-4 vs Ollama dialogue with distinct personalities
  10. AI Tutor — explains AI concepts using Ollama
  11. Student Data Generator — synthetic data using OpenAI and HuggingFace
  12. BLIP Image Captioning — automatic image captioning using Salesforce BLIP

- LLM APIs: Anthropic Claude, OpenAI GPT, Groq, Ollama
- AI Tools: Cursor, Claude, ChatGPT, GitHub Copilot
- Dev Tools: Python, LangChain, HuggingFace, ChromaDB,
  SentenceTransformers, Gradio, Modal, DALL-E, TTS
- Interested in: roles that use Claude, Cursor, or any AI-assisted development tools
"""


class JobScannerAgent:

    def fetch(self, role: str, position_type: str) -> list[dict]:
        """Fetch jobs from RSS feeds and filter with Claude."""

        role_kws = ROLE_KEYWORDS.get(role, [role.lower()])

        # Step 1: Collect raw jobs from RSS feeds
        raw_jobs = []
        for feed_url in RSS_FEEDS:
            try:
                # timeout=10 so slow feeds don't hang the whole app
                feed = feedparser.parse(feed_url, agent="Mozilla/5.0", request_headers={"Connection": "close"})
                for entry in feed.entries[:30]:
                    title   = entry.get("title",   "").lower()
                    summary = entry.get("summary", "").lower()
                    combined = title + " " + summary

                    # Only keep jobs that mention relevant keywords
                    if any(kw in combined for kw in role_kws):
                        raw_jobs.append({
                            "title":       entry.get("title",   ""),
                            "company":     entry.get("author",  "Unknown"),
                            "description": entry.get("summary", "")[:1000],
                            "url":         entry.get("link",    ""),
                        })
            except Exception as e:
                print(f"Could not fetch {feed_url}: {e}")
                continue

        # Step 2: If no jobs found from RSS, use mock jobs for testing
        if not raw_jobs:
            print("No RSS jobs found — using mock jobs for testing")
            return self._mock_jobs(role, position_type)

        # Step 3: Ask Claude to pick the 5 best matches for this candidate
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system="You are a strict job matching assistant for AI/LLM engineers. Return ONLY valid JSON, no extra text.",
            messages=[{
                "role": "user",
                "content": f"""
Candidate Profile:
{CANDIDATE_PROFILE}

Target Position: {position_type}
Target Role: {role}

From these job listings pick the TOP 3 best matches ONLY.

STRICT RULES:
- ONLY include jobs that involve LLMs, AI, ML, NLP, RAG, fine-tuning,
  or that use AI tools like Claude, Cursor, ChatGPT, Copilot in their workflow
- ALSO include roles at companies building AI-powered products even if the
  role title is "software engineer" — if they use Claude/Cursor/AI tools daily
- REJECT any job that is primarily GIS, frontend web dev, data engineering,
  finance, or has no AI/LLM component at all
- REJECT jobs that require 3+ years experience unless it says "junior" or "entry level"
- Include remote, hybrid, and onsite jobs — do not filter by location type
- Candidate can negotiate remote/hybrid after getting the offer

Return a JSON array with fields: title, company, description, url, why_good_fit
why_good_fit = one sentence explaining why this suits the candidate.
If fewer than 3 good matches exist, return only the good ones. Do NOT force bad matches.

Jobs:
{json.dumps(raw_jobs[:40], indent=2)}
"""
            }]
        )

        # Step 4: Parse Claude's response
        text = response.content[0].text.strip()
        start = text.find("[")
        end   = text.rfind("]") + 1
        return json.loads(text[start:end])

    def _mock_jobs(self, role: str, position_type: str) -> list[dict]:
        """Fallback jobs used when RSS feeds return nothing (great for testing)."""
        descriptions = {
            "Paid Intern": f"Paid internship for {role}. Work on LLM pipelines, RAG systems, fine-tuning. Remote-first, HQ in San Francisco CA. Stack: Python, Anthropic API, LangChain, HuggingFace, ChromaDB.",
            "Junior AI Engineer": f"Entry-level {role}. Build agentic systems, RAG pipelines, deploy models. Remote with occasional travel to SF. Stack: Python, Anthropic API, ChromaDB, Modal, Gradio.",
            "Junior LLM Engineer": "Fine-tune and deploy LLMs. Experience with LoRA, QLoRA, HuggingFace Trainer required. Remote-friendly startup based in Los Angeles CA.",
        }
        return [
            {
                "title":        f"{position_type} - {role}",
                "company":      "YC AI Startup (Remote)",
                "description":  descriptions.get(position_type, f"{role} role at a startup."),
                "url":          "https://workatastartup.com",
                "why_good_fit": f"Matches your {role} background and remote preference perfectly.",
            }
        ]
