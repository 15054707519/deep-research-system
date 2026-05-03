import asyncio
from llm_client import LLMClient
from router import Router
from parallel import ParallelProcessor
from reflection import ReflectionLoop

async def main():
    producer_llm = LLMClient(default_model="qwen2.5:7b")
    critic_llm = LLMClient(default_model="qwen2.5:7b")
    
    router = Router(producer_llm)
    query = "What are the main bottlenecks of current quantum computing?"
    print(f"User question: {query}")
    
    route_result = await router.route(query)
    domain = route_result.get("domain")
    if domain == "reject":
        print("Question rejected.")
        return
    print(f"Routed domain: {domain}")

    parallel = ParallelProcessor(producer_llm, domain, n_candidates=3)
    initial_report = await parallel.run(query)
    print("\n" + "="*60)
    print("📄 Initial Research Report")
    print("="*60)
    print(initial_report)

    reflector = ReflectionLoop(producer_llm, critic_llm, max_iter=3, threshold=7.0)
    final_report = await reflector.refine(initial_report, query, domain)
    
    print("\n" + "="*60)
    print("🎯 Final Research Report (after reflection)")
    print("="*60)
    print(final_report)

if __name__ == "__main__":
    asyncio.run(main())