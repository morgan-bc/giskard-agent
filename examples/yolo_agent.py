import os
import asyncio
import json
from pathlib import Path
from dotenv import load_dotenv
from giskard import create_harness_agent, Agent
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
    
    agent: Agent = create_harness_agent(
        name="yolo_agent",
        client=client,
        workdir=WORKDIR,
        tool_approval_rule="yolo",
        web_search_client=parallel_client,
        disable_file_memory=True,
        disable_todo=True,
    )    
    session = agent.create_session()
    
    # query = "调研 agent harness 持续学习的研究，并将结果保存到工作目录下"
    # query = "调研当前AI算力中内存需求情况，并将结果保存到工作目录下"
    # pending = None
    # async for update in agent.run(query, stream=True, session=session):
    #     if update.text:
    #         print(update.text, end='', flush=True)
    #         pending = None
    #     elif update.contents:
    #         content = update.contents[0]
    #         if pending is None:
    #             pending = content
    #         elif pending.type != content.type:
    #             print(f'\n{pending.to_dict()}', flush=True)
    #             pending = None
    #         else:
    #             pending += content            

    # query = "调研并分析比亚迪2026年上半年的销售数据，最后将结果保存到工作目录下"
    # query = "调用中国平安基本面数据，并将结果保存到工作目录下"
    # query = "调用 glob 工具查看当前工作目录有哪些文件"
    # query = "调研赣锋锂业基本面数据，并将结果保存到工作目录下"
    query = "According to wikipedia, how many Asian countries still have a monarchy and access to the sea in 2021?"
    stream = await agent.run(query, session=session, stream=True)
    final_response = await stream.get_final_response()
    # final_messages = final_response
    messages = [msg.to_dict() for msg in final_response.messages]

    with open(WORKDIR / "messages.json", "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)
    
    print(final_response.text)
    last_invocation = session.last_invocation or {}
    trajectory = {
        "system_prompt": last_invocation.get("system_prompt"),
        "tools": last_invocation.get("tools", []),
        "messages": [msg.to_dict() for msg in session.state["in_memory"]["messages"]],
    }
    with open(WORKDIR / "trajectory.json", "w", encoding="utf-8") as f:
        json.dump(trajectory, f, ensure_ascii=False, indent=2)

    await parallel_client.close()            

if __name__ == "__main__":
    asyncio.run(main())
