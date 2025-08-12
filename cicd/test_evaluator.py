import pytest
import uuid
import json

from langgraph.types import Command
from langchain_openai import ChatOpenAI
from openevals.llm import create_async_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT
from langsmith import Client
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=True)

from agent.multiagent import multiagent

client = Client()

model = ChatOpenAI(model="o3-mini")

# Using Open Eval pre-built 
correctness_evaluator = create_async_llm_as_judge(
    prompt=CORRECTNESS_PROMPT,
    feedback_key="correctness",
    judge=model,
)

async def run_graph(inputs: dict):
    """Run graph and track the final response."""
    # Creating configuration 
    graph = multiagent
    thread_id = uuid.uuid4()
    configuration = {"thread_id": thread_id, "user_id" : "10"}

    # Invoke graph until interrupt 
    result = await graph.ainvoke(inputs, config = configuration)

    # Proceed from human-in-the-loop 
    result = await graph.ainvoke(Command(resume="My customer ID is 10"), config={"thread_id": thread_id, "user_id" : "10"})
    
    return {"messages": [{"role": "ai", "content": result['messages'][-1].content}]}


@pytest.mark.evaluator
@pytest.mark.asyncio
async def test_evaluate_graph(dataset_name = "LangGraph 101 Multi-Agent: Final Response"):
    # Evaluation job and results
    experiment_results = await client.aevaluate(
        run_graph,
        data=dataset_name,
        evaluators=[correctness_evaluator],
        experiment_prefix="agent-o3mini-e2e",
        num_repetitions=1,
        max_concurrency=5,
    )

    assert experiment_results is not None
    print(f"✅ Evaluation completed: {experiment_results.experiment_name}")

    # Define scoring rules
    criteria = {"correctness": ">=0.75"}

    output_metadata = {
        "experiment_name": experiment_results.experiment_name,
        "criteria": criteria,
    }

    safe_name = experiment_results.experiment_name.replace(":", "-").replace("/", "-")
    config_filename = f"evaluation_config__{safe_name}.json"
    with open(config_filename, "w") as f:
        json.dump(output_metadata, f)

    print(f"::set-output name=config_filename::{config_filename}")


