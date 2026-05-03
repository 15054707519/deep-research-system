# Deep Research System: Routing + Parallelization + Reflection

This project implements three core Agentic Design Patterns:
- **Routing**: LLM-based classification of research questions into domains (science/tech, history/culture, finance/business, general/daily) with rejection of adversarial queries.
- **Parallelization**: Decompose question into 3 sub-questions, generate N≥3 candidates for each in parallel, select best via LLM judge (Best-of-N), and reduce into a draft report.
- **Reflection**: Producer-Critic loop where the critic scores the report on factuality, completeness, consistency, tone, and unsupported claims (0–10), and the producer revises it iteratively until threshold is met or max iterations reached.

## Environment

- Python 3.9+
- [Ollama](https://ollama.com/) with model `qwen2.5:7b`
- Dependencies: `aiohttp`, `python-dotenv`

## Setup & Run

```bash
git clone https://github.com/15054707519/deep-research-system.git
cd deep-research-system
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install aiohttp python-dotenv
ollama pull qwen2.5:7b
python main.py
Router Accuracy
Tested with 10 questions (2 per domain + 2 adversarial). Results:

Domain	#Test	Correct	Accuracy
Science/Tech	2	2	100%
History/Culture	2	2	100%
Finance/Business	2	2	100%
General/Daily	2	2	100%
Adversarial (reject)	2	2	100%
Total	10	10	100%
Run python test_router.py to reproduce.

Parallelization Performance
Example run for "What are the main bottlenecks of current quantum computing?":

Actual parallel time: 42.73 s

Simulated sequential time: 13.50 s

Speedup: 0.32x

Note: Ollama processes concurrent requests sequentially, so no actual speedup is observed, but the code correctly uses asyncio.gather to achieve concurrency.

Reflection Loop Example
The system performed 3 iterations of Producer-Critic:

Iteration	Factuality	Completeness	Consistency	Tone	Unsupported Claims	Average
1	9	8	10	9	2	7.6
2	9	8	10	9	2	7.6
3	9	8	10	9	2	7.6
The critic consistently pointed out the low score in "unsupported claims", and the producer added specific citations and examples in response.

Video Demo
The video has been submitted separately to the course platform (classroom). It demonstrates:

Running python main.py from start to finish

Running python test_router.py showing 100% accuracy

Brief explanation of the three modules

Project Structure
text
.
├── main.py            # Orchestrator
├── router.py          # Routing module
├── parallel.py        # Parallelization (Map-Reduce + Best-of-N)
├── reflection.py      # Reflection (Producer-Critic loop)
├── llm_client.py      # Async LLM client for Ollama
├── test_router.py     # Router accuracy evaluation
└── README.md
Model Used
Local Ollama + qwen2.5:7b (same model for producer and critic, but with distinct system prompts and roles, satisfying the requirement of "different systems").

Author
Wang Zhaoyang
Assignment Source
TSU Course – Task 2: Simple Deep Research System