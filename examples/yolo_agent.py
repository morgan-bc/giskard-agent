import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from giskard import create_harness_agent
from giskard.providers import OpenAIChatCompletionClient
from giskard.core.types import Content
from giskard.tools.web_search import ParallelSearchClient


load_dotenv()

WORKDIR = Path(__file__).resolve().parents[1] / "workdir"

async def main():
    client = OpenAIChatCompletionClient(
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        model=os.getenv("BASE_MODEL"),
    )
    
    parallel_client = ParallelSearchClient()
    
    agent = create_harness_agent(
        name="yolo_agent",
        client=client,
        workdir=WORKDIR,
        tool_approval_rule="yolo",
        web_search_client=parallel_client,
    )    
    session = agent.create_session()
    
    # query = "调研 agent harness 持续学习的研究，并将结果保存到工作目录下"
    query = "调研当前AI算力中内存需求情况，并将结果保存到工作目录下"
    pending = None
    async for update in agent.run(query, stream=True, session=session):
        if update.text:
            print(update.text, end='', flush=True)
            pending = None
        elif update.contents:
            content = update.contents[0]
            if pending is None:
                pending = content
            elif pending.type != content.type:
                print(f'\n{pending.to_dict()}', flush=True)
                pending = None
            else:
                pending += content            

    await parallel_client.close()            

if __name__ == "__main__":
    asyncio.run(main())
