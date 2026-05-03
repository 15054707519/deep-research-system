import json
import difflib
from llm_client import LLMClient

class ReflectionLoop:
    def __init__(self, producer_llm: LLMClient, critic_llm: LLMClient, max_iter=3, threshold=7.0):
        self.producer = producer_llm
        self.critic = critic_llm
        self.max_iter = max_iter
        self.threshold = threshold

    async def refine(self, initial_draft: str, query: str, domain: str) -> str:
        draft = initial_draft
        prev_draft = ""
        
        print(f"\n{'='*60}")
        print("Start Reflection Loop (Producer-Critic mode)")
        print(f"Max iterations: {self.max_iter}, Score threshold: {self.threshold}")
        print(f"{'='*60}")
        
        for i in range(self.max_iter):
            print(f"\n--- Iteration {i+1} ---")
            
            critique = await self._critique(draft, query, domain)
            scores = critique.get("scores", {})
            suggestions = critique.get("suggestions", "Please improve the answer quality.")
            
            print(f"\n📊 Critic Scores:")
            for dim, score in scores.items():
                print(f"   {dim}: {score}/10")
            
            avg_score = sum(scores.values()) / len(scores) if scores else 0
            print(f"   📈 Average: {avg_score:.1f}/10")
            
            if all(v >= self.threshold for v in scores.values()):
                print(f"\n✅ All dimensions meet threshold {self.threshold}, stopping early")
                break
            
            if prev_draft:
                diff_lines = difflib.unified_diff(
                    prev_draft.splitlines(), 
                    draft.splitlines(), 
                    lineterm='',
                    fromfile='previous',
                    tofile='current'
                )
                diff_text = list(diff_lines)
                if diff_text:
                    print(f"\n📝 Text changes (first 10 lines):")
                    for line in diff_text[:10]:
                        print(f"   {line}")
            
            print(f"\n🔧 Producer revising based on critic feedback...")
            new_draft = await self._revise(draft, suggestions, query, domain)
            prev_draft = draft
            draft = new_draft
        
        print(f"\n{'='*60}")
        print("Reflection loop finished")
        print(f"{'='*60}")
        return draft

    async def _critique(self, draft: str, query: str, domain: str) -> dict:
        prompt = f"""You are a strict research report evaluator. Rate the following report on five dimensions (0-10):

Dimensions:
- factuality: whether the content is based on facts/evidence, no fabrication
- completeness: covers key aspects of the question
- consistency: internal logic, no contradictions
- tone: appropriate for a {domain} research report
- unsupported_claims: presence of claims without evidence (lower score means more problems)

Output strict JSON format:
{{"scores": {{"factuality": 0-10, "completeness": 0-10, "consistency": 0-10, "tone": 0-10, "unsupported_claims": 0-10}}, "suggestions": "specific actionable feedback"}}

Original user question: {query}
Domain: {domain}
Report:
{draft}
"""
        resp = await self.critic.complete(prompt, temperature=0.3)
        try:
            start = resp.find('{')
            end = resp.rfind('}') + 1
            if start != -1 and end > start:
                json_str = resp[start:end]
                data = json.loads(json_str)
                return data
        except:
            pass
        return {
            "scores": {
                "factuality": 6,
                "completeness": 6,
                "consistency": 6,
                "tone": 6,
                "unsupported_claims": 6
            },
            "suggestions": "Please provide more factual evidence and ensure completeness and coherence."
        }

    async def _revise(self, draft: str, suggestions: str, query: str, domain: str) -> str:
        prompt = f"""You are a {domain} research report writer. Revise the report according to the following critique.

Original question: {query}
Critique suggestions: {suggestions}

Current report:
{draft}

Output the revised complete report (only the report, no extra comments):
"""
        return await self.producer.complete(prompt, temperature=0.6)