import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import TaskCompletionMetric
from deepeval.tracing import observe, update_current_trace
from judge import GroqJudge

dataset = EvaluationDataset(goldens=[Golden(input="Plan a three-day trip to Paris")])

@observe()
async def my_ai_agent(query: str) -> str:
    answer = "..."  # your LLM call here
    update_current_trace(input=query, output=answer)
    return answer

judge = GroqJudge()

for golden in dataset.evals_iterator(metrics=[TaskCompletionMetric(model=judge)]):
    task = asyncio.create_task(my_ai_agent(golden.input))
    dataset.evaluate(task)