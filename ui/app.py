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


def build_html_table(unique):
    """Build an HTML table with clickable Apply links."""
    rows_html = ""
    for r in unique:
        salary     = r["salary"]
        period     = salary.get("period", "annual")
        gaps       = ", ".join(r["skill_gaps"][:3]) if r["skill_gaps"] else "None"
        course     = r["courses"][0]["title"] if r["courses"] else "—"
        why        = r["job"].get("why_good_fit", "—")
        salary_str = f"${salary['min']:,} – ${salary['max']:,} ({period})"
        url        = r["job"].get("url", "")
        apply_btn  = f'<a href="{url}" target="_blank" style="background:#f97316;color:white;padding:6px 12px;border-radius:6px;text-decoration:none;font-weight:bold;">Apply</a>' if url else "—"

        rows_html += f"""
        <tr style="border-bottom:1px solid #374151">
            <td style="padding:10px;color:#f9fafb">{r["position_type"]}</td>
            <td style="padding:10px;color:#f9fafb">{r["target_role"]}</td>
            <td style="padding:10px;color:#f9fafb;font-weight:bold">{r["job"]["title"]}</td>
            <td style="padding:10px;color:#f9fafb">{r["job"]["company"]}</td>
            <td style="padding:10px;color:#f9fafb">{salary_str}</td>
            <td style="padding:10px;color:#34d399;text-align:center;font-weight:bold">{salary["fit_score"]}/100</td>
            <td style="padding:10px;color:#d1d5db;font-size:0.85em">{gaps}</td>
            <td style="padding:10px;color:#d1d5db;font-size:0.85em">{course}</td>
            <td style="padding:10px;color:#d1d5db;font-size:0.85em">{why}</td>
            <td style="padding:10px;text-align:center">{apply_btn}</td>
        </tr>"""

    return f"""
    <div style="overflow-x:auto;border-radius:10px;overflow:hidden">
    <table style="width:100%;border-collapse:collapse;font-size:0.9em;background:#111827">
        <thead>
            <tr style="background:#1f2937;text-align:left">
                <th style="padding:12px;color:#9ca3af;font-size:0.8em;text-transform:uppercase">Position</th>
                <th style="padding:12px;color:#9ca3af;font-size:0.8em;text-transform:uppercase">Role</th>
                <th style="padding:12px;color:#9ca3af;font-size:0.8em;text-transform:uppercase">Job Title</th>
                <th style="padding:12px;color:#9ca3af;font-size:0.8em;text-transform:uppercase">Company</th>
                <th style="padding:12px;color:#9ca3af;font-size:0.8em;text-transform:uppercase">Salary</th>
                <th style="padding:12px;color:#9ca3af;font-size:0.8em;text-transform:uppercase">Fit Score</th>
                <th style="padding:12px;color:#9ca3af;font-size:0.8em;text-transform:uppercase">Skill Gaps</th>
                <th style="padding:12px;color:#9ca3af;font-size:0.8em;text-transform:uppercase">Top Course</th>
                <th style="padding:12px;color:#9ca3af;font-size:0.8em;text-transform:uppercase">Why Good Fit</th>
                <th style="padding:12px;color:#9ca3af;font-size:0.8em;text-transform:uppercase">Apply</th>
            </tr>
        </thead>
        <tbody>{rows_html}</tbody>
    </table>
    </div>"""


def run_coach(resume_file, roles, position_types, custom_role):
    """Main function — runs when user clicks the button."""

    # Validate inputs
    if resume_file is None:
        return "", "⚠️ Please upload your resume PDF."
    if not roles:
        return "", "⚠️ Please select at least one role."
    if not position_types:
        return "", "⚠️ Please select at least one position type."

    # Add custom role if typed in
    target_roles = list(roles)
    if custom_role.strip():
        target_roles.append(custom_role.strip())

    try:
        # Step 1: Read resume PDF and store in ChromaDB
        print("📄 Reading resume...")
        ingest_resume(resume_file.name)
    except Exception as e:
        return "", f"❌ Error reading resume: {str(e)}"

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
        job_url = r["job"].get("url", "").lower().strip()
        if job_url not in seen:
            seen.add(job_url)
            unique.append(r)

    return build_html_table(unique), f"✅ Done! Found {len(unique)} opportunities."


# Build the UI
with gr.Blocks(title="AI Career Coach") as app:

    gr.Markdown("""
# AI Career Coach Agent
Powered by Claude Sonnet 4.6 + RAG + Multi-Agent System
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
            btn = gr.Button("Find Opportunities", variant="primary", size="lg")

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

    table = gr.HTML()

    # Wire button to function
    btn.click(
        fn=run_coach,
        inputs=[resume, roles, position_types, custom_role],
        outputs=[table, status]
    )

if __name__ == "__main__":
    app.launch()
