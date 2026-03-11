import sys
import os

# Make sure Python can find our agents/ and rag/ folders
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
from agents.planner_agent import PlannerAgent
from rag.ingest import ingest_resume
from concurrent.futures import ThreadPoolExecutor, as_completed

# Create planner once — not every time button is clicked
planner = PlannerAgent()

POSITION_TYPES = ["Paid Intern", "Junior AI Engineer", "Junior LLM Engineer"]
ROLES          = ["AI Engineer", "LLM Engineer"]


def run_coach(resume_file, roles, position_types, custom_role):
    """Main function — runs when user clicks the button."""

    # Validate inputs
    if resume_file is None:
        return [], "⚠️ Please upload your resume PDF."
    if not roles:
        return [], "⚠️ Please select at least one role."
    if not position_types:
        return [], "⚠️ Please select at least one position type."

    # Add custom role if typed in
    target_roles = list(roles)
    if custom_role.strip():
        target_roles.append(custom_role.strip())

    try:
        # Step 1: Read resume PDF and store in ChromaDB
        print("📄 Reading resume...")
        ingest_resume(resume_file.name)
    except Exception as e:
        return [], f"❌ Error reading resume: {str(e)}"

    all_results = []

    # Step 2: Run ALL role + position combinations in parallel
    def search(role, position_type):
        print(f"🔍 Searching: {position_type} - {role}")
        results = planner.run(resume_file.name, role, position_type)
        for r in results:
            r["target_role"]   = role
            r["position_type"] = position_type
        return results

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(search, role, pt): (role, pt)
            for role in target_roles
            for pt in position_types
        }
        try:
            # Wait up to 3 minutes for all searches
            for future in as_completed(futures, timeout=180):
                role, pt = futures[future]
                try:
                    all_results.extend(future.result())
                    print(f"✅ Done: {pt} - {role}")
                except Exception as e:
                    print(f"⚠️ Skipped {pt} - {role}: {e}")
        except Exception:
            # Timeout hit — collect whatever finished so far
            print("⏰ Timeout — collecting finished results...")
            for future, (role, pt) in futures.items():
                if future.done():
                    try:
                        all_results.extend(future.result())
                    except Exception:
                        pass

    # Step 3: Remove duplicate jobs, sort by fit score
    seen   = set()
    unique = []
    for r in sorted(all_results, key=lambda x: x["salary"]["fit_score"], reverse=True):
        url = r["job"].get("url", "")
        if url not in seen:
            seen.add(url)
            unique.append(r)

    # Step 4: Format into table rows
    rows = []
    for r in unique:
        salary     = r["salary"]
        period     = salary.get("period", "annual")
        gaps       = ", ".join(r["skill_gaps"][:3]) if r["skill_gaps"] else "None"
        course     = r["courses"][0]["title"] if r["courses"] else "—"
        why        = r["job"].get("why_good_fit", "—")
        salary_str = f"${salary['min']:,} – ${salary['max']:,} ({period})"

        rows.append([
            r["position_type"],
            r["target_role"],
            r["job"]["title"],
            r["job"]["company"],
            salary_str,
            f"{salary['fit_score']}/100",
            gaps,
            course,
            why,
        ])

    return rows, f"✅ Done! Found {len(rows)} opportunities."


# Build the UI
with gr.Blocks(title="AI Career Coach") as app:

    gr.Markdown("""
# 🤖 AI Career Coach Agent
### Powered by Claude Sonnet 4.6 + RAG + Multi-Agent System
**Finds remote startup jobs · Spots skill gaps · Recommends free courses · Estimates salary**
""")

    with gr.Row():

        # Left column — inputs
        with gr.Column(scale=1):
            resume = gr.File(
                label="📄 Upload Resume (PDF)",
                file_types=[".pdf"]
            )
            position_types = gr.CheckboxGroup(
                choices=POSITION_TYPES,
                value=POSITION_TYPES,
                label="Position Types"
            )
            roles = gr.CheckboxGroup(
                choices=ROLES,
                value=ROLES,
                label="Target Roles"
            )
            custom_role = gr.Textbox(
                label="Add Custom Role (optional)",
                placeholder="e.g. ML Engineer, Prompt Engineer"
            )
            btn = gr.Button("🔍 Find Opportunities", variant="primary", size="lg")

        # Right column — status
        with gr.Column(scale=1):
            gr.Markdown("""
### How it works
1. Upload your resume PDF
2. Select position types and roles
3. Click Find Opportunities
4. The system will:
   - Scan job listings from startup sites
   - Compare each job to your resume
   - Find skill gaps
   - Recommend free courses
   - Estimate salary ranges
   - Show fit score for each job
""")

    status = gr.Textbox(label="Status", interactive=False)

    table = gr.Dataframe(
        headers=[
            "Position", "Role", "Job Title", "Company",
            "Salary", "Fit Score", "Skill Gaps", "Top Course", "Why Good Fit"
        ],
        interactive=False,
        wrap=True
    )

    # Wire button to function
    btn.click(
        fn=run_coach,
        inputs=[resume, roles, position_types, custom_role],
        outputs=[table, status]
    )

if __name__ == "__main__":
    app.launch()
