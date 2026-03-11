# 🤖 AI Career Coach Agent

An agentic AI system that finds remote startup jobs, spots skill gaps, recommends free courses, and estimates salary — powered by Claude Sonnet 4.6.

## Demo
Upload your resume → get matched to AI/LLM jobs at startups → see skill gaps + free courses + salary estimates

## Architecture

```
rag/
  embedder.py       → converts text to vectors (SentenceTransformers)
  vector_store.py   → stores and searches resume chunks (ChromaDB)
  ingest.py         → reads PDF resume and stores in ChromaDB

agents/
  job_scanner_agent.py  → finds jobs from RSS feeds, Claude filters best matches
  skill_gap_agent.py    → RAG + Claude compares resume to job requirements
  learning_agent.py     → Claude recommends free courses for each gap
  salary_agent.py       → Claude estimates salary + fit score (0-100)
  alert_agent.py        → sends top matches via Pushover notification
  planner_agent.py      → orchestrates all agents in order

ui/
  app.py            → Gradio web interface
```

## Tech Stack
- **Claude Sonnet 4.6** — skill gap analysis, job filtering, salary estimation
- **ChromaDB** — vector store for resume chunks
- **SentenceTransformers** — local embeddings (free, no API cost)
- **Gradio** — web UI
- **feedparser** — RSS job feeds (RemoteOK, WeWorkRemotely)
- **Pushover** — phone notifications (optional)

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/ai-career-coach-agent
cd ai-career-coach-agent

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API key
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# 4. Run the app
python3 ui/app.py

# 5. Open browser
# http://localhost:7860
```

## How It Works

1. Upload your resume PDF
2. Select position types (Paid Intern / Junior AI Engineer / Junior LLM Engineer)
3. Select target roles (AI Engineer / LLM Engineer)
4. Click **Find Opportunities**

The system will:
- Scan job listings from AI/ML startup job boards
- Compare each job to your resume using RAG
- Find skill gaps (only real gaps — not things you already know)
- Recommend free courses for each gap
- Estimate salary range + fit score (0-100)
- Show results sorted by fit score

## Target Jobs
- Remote-first startups (California-based or fully remote)
- Companies using Claude, Cursor, ChatGPT in their workflow
- Positions: Paid Intern, Junior AI Engineer, Junior LLM Engineer

## Built With
This project was built as a Week 8 capstone for Ed Donner's LLM Engineering course.
