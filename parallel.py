import asyncio
import json
import time
from typing import List
from llm_client import LLMClient

class ParallelProcessor:
    def __init__(self, llm: LLMClient, domain: str, n_candidates: int = 3):
        self.llm = llm
        self.domain = domain
        self.n_candidates = n_candidates

    async def run(self, query: str) -> str:
        sub_questions = await self._decompose(query)
        print(f"\n[Parallel] Sub-questions: {sub_questions}")

        start_time = time.time()
        best_answers = await asyncio.gather(*[self._process_subq(q) for q in sub_questions])
        parallel_time = time.time() - start_time

        simulated_sequential = len(sub_questions) * self.n_candidates * 1.5
        print(f"\n[Performance] Actual parallel time: {parallel_time:.2f} sec")
        print(f"[Performance] Simulated sequential time: {simulated_sequential:.2f} sec")
        print(f"[Performance] Speedup: {simulated_sequential/parallel_time:.2f}x")

        final_report = await self._reduce(query, sub_questions, best_answers)
        return final_report

    async def _decompose(self, query: str) -> List[str]:
        prompt = f"""You are a {self.domain} expert. Decompose the following question into 3 independent sub-questions that can be answered in parallel.
Each sub-question must be specific and not dependent on others. Output a JSON array, e.g. ["subQ1", "subQ2", "subQ3"].

Question: {query}
"""
        resp = await self.llm.complete(prompt, temperature=0.5)
        try:
            subqs = json.loads(resp)
            if isinstance(subqs, list) and len(subqs) == 3:
                return subqs
        except:
            pass
        return [f"{query} - part 1", f"{query} - part 2", f"{query} - part 3"]

    async def _process_subq(self, sub_q: str) -> str:
        candidates = await asyncio.gather(*[self._generate_candidate(sub_q) for _ in range(self.n_candidates)])
        best = await self._select_best(sub_q, candidates)
        return best

    async def _generate_candidate(self, sub_q: str) -> str:
        prompt = f"You are a {self.domain} expert. Answer directly and concisely: {sub_q}"
        return await self.llm.complete(prompt, temperature=0.8)

    async def _select_best(self, sub_q: str, candidates: List[str]) -> str:
        scores = []
        for cand in candidates:
            prompt = f"""Score the following answer from 0 to 10 based on:
- Correctness (fact-based)
- Relevance (directly answers the question)
- Completeness (sufficient)

Question: {sub_q}
Answer: {cand}

Output JSON: {{"score": number}}
"""
            resp = await self.llm.complete(prompt, temperature=0.2)
            try:
                score = json.loads(resp).get("score", 5)
            except:
                score = 5
            scores.append(score)
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        print(f"  Best candidate score for '{sub_q[:30]}...': {scores[best_idx]}")
        return candidates[best_idx]

    async def _reduce(self, query: str, sub_qs: List[str], answers: List[str]) -> str:
        combined = "\n\n".join([f"## Sub-question {i+1}: {sub_qs[i]}\n{answers[i]}" for i in range(3)])
        prompt = f"""You are a {self.domain} expert. Based on the answers to the three sub-questions, produce a coherent research report that directly addresses the original question.

Original question: {query}

{combined}

Research report:
"""
        return await self.llm.complete(prompt, temperature=0.5)