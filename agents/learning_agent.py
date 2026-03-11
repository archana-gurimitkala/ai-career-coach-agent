import anthropic
import json
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()


class LearningAgent:

    def recommend(self, skill_gaps: list[str]) -> list[dict]:
        """Recommend one free course per skill gap."""

        # If no gaps found, nothing to recommend
        if not skill_gaps:
            return []

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system="You are a learning advisor for early-career AI engineers. Return ONLY valid JSON, no extra text.",
            messages=[{
                "role": "user",
                "content": f"""
For each of these skills, recommend ONE free online course or resource.
Prefer: YouTube tutorials, official docs, free Coursera/edX, freeCodeCamp.

Skills to cover: {skill_gaps}

Return a JSON array like this:
[
  {{
    "skill": "Docker",
    "title": "Docker Tutorial for Beginners",
    "platform": "YouTube - TechWorld with Nana",
    "duration": "3 hours",
    "url": "https://www.youtube.com/watch?v=3c-iBn73dDE"
  }}
]

Return ONLY the JSON array, nothing else.
"""
            }]
        )

        # Parse Claude's response
        text  = response.content[0].text.strip()
        start = text.find("[")
        end   = text.rfind("]") + 1

        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            print(f"Could not parse courses: {text}")
            return []
