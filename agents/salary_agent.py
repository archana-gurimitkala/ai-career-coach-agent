import anthropic
import json
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()

# Salary context per position type
# Helps Claude give realistic ranges not generic ones
POSITION_SALARY_CONTEXT = {
    "Paid Intern":          "Paid internship. Typical US startup pay: $25-45/hr or $4,000-8,000/month.",
    "Junior AI Engineer":   "Junior/entry-level role. Typical US startup salary: $80,000-120,000/year.",
    "Junior LLM Engineer":  "Junior/entry-level LLM role. Typical US startup salary: $90,000-130,000/year.",
}


class SalaryAgent:

    def estimate(self, job: dict, position_type: str = "Junior AI Engineer") -> dict:
        """Estimate salary range and fit score for a job."""

        salary_context = POSITION_SALARY_CONTEXT.get(position_type, "")

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            # Adaptive thinking = Claude reasons carefully before answering
            # Great for salary estimation which needs multi-step reasoning
            thinking={"type": "adaptive"},
            system="You are a compensation expert for US startup jobs. Return ONLY valid JSON, no extra text.",
            messages=[{
                "role": "user",
                "content": f"""
Estimate compensation for this position.

Position Type: {position_type}
Salary Reference: {salary_context}

Job Title: {job.get('title', 'Unknown')}
Company:   {job.get('company', 'Unknown')}
Description: {job.get('description', '')[:500]}

Notes:
- Remote role, startup company
- California market rates if company is based there
- Candidate has LLM, RAG, fine-tuning background

Return JSON exactly like this:
{{
  "min": 90000,
  "max": 120000,
  "currency": "USD",
  "period": "annual",
  "fit_score": 82
}}

fit_score = how well an early-career candidate with LLM/RAG/fine-tuning background
matches this role (0-100). Be realistic and honest.
"""
            }]
        )

        # Claude with adaptive thinking returns multiple content blocks
        # We skip thinking blocks and only read the text block
        for block in response.content:
            if block.type == "text":
                text  = block.text.strip()
                start = text.find("{")
                end   = text.rfind("}") + 1
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    print(f"Could not parse salary: {text}")

        # Fallback if parsing fails
        return {"min": 0, "max": 0, "currency": "USD", "period": "annual", "fit_score": 0}
