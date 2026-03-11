from agents.job_scanner_agent import JobScannerAgent
from agents.skill_gap_agent   import SkillGapAgent
from agents.learning_agent    import LearningAgent
from agents.salary_agent      import SalaryAgent
from agents.alert_agent       import AlertAgent


class PlannerAgent:

    def __init__(self):
        # Create all agents once on startup
        self.scanner   = JobScannerAgent()
        self.skill_gap = SkillGapAgent()
        self.learning  = LearningAgent()
        self.salary    = SalaryAgent()
        self.alert     = AlertAgent()

    def run(self, resume_path: str, role: str, position_type: str) -> list[dict]:
        """Run all agents in order and return results."""

        print(f"\n🔍 Scanning jobs for: {position_type} - {role}")

        # Step 1: Find relevant jobs
        jobs = self.scanner.fetch(role, position_type)
        print(f"   Found {len(jobs)} jobs")

        results = []

        # Step 2: Analyze each job
        for i, job in enumerate(jobs[:5], 1):
            print(f"   Analyzing job {i}/{min(len(jobs), 5)}: {job['title']} @ {job['company']}")

            # Find skill gaps using RAG + Claude
            gaps = self.skill_gap.analyze(
                job,
                role=role,
                position_type=position_type
            )

            # Recommend free courses for each gap
            courses = self.learning.recommend(gaps)

            # Estimate salary and fit score
            salary = self.salary.estimate(job, position_type=position_type)

            # Collect everything together
            results.append({
                "job":           job,
                "skill_gaps":    gaps,
                "courses":       courses,
                "salary":        salary,
                "target_role":   role,
                "position_type": position_type,
            })

        # Step 3: Sort by fit score — best match first
        results.sort(key=lambda x: x["salary"]["fit_score"], reverse=True)

        # Step 4: Notify about top 3
        self.alert.notify(results[:3])

        return results
