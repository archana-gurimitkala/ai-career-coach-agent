import os
import requests
from dotenv import load_dotenv

load_dotenv()


class AlertAgent:

    def __init__(self):
        # These come from .env file
        # If not set, Pushover is skipped and we print instead
        self.token = os.getenv("PUSHOVER_TOKEN")
        self.user  = os.getenv("PUSHOVER_USER")

    def notify(self, top_results: list[dict]):
        """Send top job matches as a notification."""

        if not top_results:
            print("No results to notify about.")
            return

        # Build the message text
        message = self._build_message(top_results)

        # If Pushover is configured → send to phone
        if self.token and self.user:
            self._send_pushover(message)
        else:
            # Otherwise just print to terminal
            self._print_results(top_results)

    def _build_message(self, top_results: list[dict]) -> str:
        """Format the top jobs into a readable message."""
        lines = ["Top Job Matches:\n"]

        for i, r in enumerate(top_results, 1):
            job      = r["job"]
            salary   = r["salary"]
            position = r.get("position_type", "")

            lines.append(f"{i}. {job['title']} @ {job['company']}")
            lines.append(f"   Position: {position}")
            lines.append(f"   Salary:   ${salary['min']:,} - ${salary['max']:,} ({salary.get('period','annual')})")
            lines.append(f"   Fit Score: {salary['fit_score']}/100")
            lines.append(f"   URL: {job.get('url', 'N/A')}")
            lines.append("")

        return "\n".join(lines)

    def _send_pushover(self, message: str):
        """Send notification via Pushover API."""
        try:
            response = requests.post(
                "https://api.pushover.net/1/messages.json",
                data={
                    "token":   self.token,
                    "user":    self.user,
                    "title":   "AI Career Coach - Top Matches",
                    "message": message,
                }
            )
            if response.status_code == 200:
                print("✅ Notification sent to your phone!")
            else:
                print(f"Pushover error: {response.text}")
        except Exception as e:
            print(f"Could not send notification: {e}")

    def _print_results(self, top_results: list[dict]):
        """Print results to terminal when Pushover is not configured."""
        print("\n" + "="*50)
        print("TOP JOB MATCHES")
        print("="*50)

        for i, r in enumerate(top_results, 1):
            job    = r["job"]
            salary = r["salary"]
            gaps   = r.get("skill_gaps", [])
            course = r.get("courses", [{}])

            print(f"\n{i}. {job['title']} @ {job['company']}")
            print(f"   Position:  {r.get('position_type', '')}")
            print(f"   Salary:    ${salary['min']:,} - ${salary['max']:,} ({salary.get('period','annual')})")
            print(f"   Fit Score: {salary['fit_score']}/100")
            print(f"   Gaps:      {', '.join(gaps[:3]) if gaps else 'None'}")
            if course and course[0]:
                print(f"   Learn:     {course[0].get('title','')}")
            print(f"   URL:       {job.get('url','N/A')}")

        print("="*50 + "\n")
