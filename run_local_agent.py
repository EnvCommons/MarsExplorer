"""Local test agent that runs against the MarsExplorer environment directly,
logs all interactions to a .jsonl trajectory file, and prints a summary."""

import asyncio
import json
import os
import time
from datetime import datetime, timezone

from dotenv import load_dotenv
from openai import AsyncOpenAI

from marsexplorer import MarsExplorerEnvironment, MoveParams

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

TOOLS = [
    {
        "type": "function",
        "name": "move",
        "description": "Move the rover one step in the given direction (up/down/left/right). Returns the updated map.",
        "parameters": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["up", "down", "left", "right"],
                    "description": "Direction to move the rover.",
                }
            },
            "required": ["direction"],
            "additionalProperties": False,
        },
        "strict": True,
    }
]


async def run_task(oai_client, task_spec, model_name, trajectory_file):
    env = MarsExplorerEnvironment(task_spec)
    await env.setup()

    prompt = await env.get_prompt()
    prompt_text = prompt[0].text

    input_list = [{"role": "user", "content": prompt_text}]
    finished = False
    step = 0
    trajectory = []

    trajectory.append({
        "type": "prompt",
        "task_id": task_spec["id"],
        "difficulty": task_spec["difficulty"],
        "grid_size": task_spec["grid_size"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    start_time = time.time()

    while not finished:
        response = await oai_client.responses.create(
            model=model_name,
            tools=TOOLS,
            input=input_list,
        )

        input_list += response.output

        for item in response.output:
            if item.type == "function_call":
                step += 1
                args = json.loads(str(item.arguments))

                result = await env.move(MoveParams(**args))

                reward = result.reward
                finished = result.finished
                output_text = result.blocks[0].text

                entry = {
                    "type": "step",
                    "step": step,
                    "direction": args.get("direction"),
                    "position": [env.agent_x, env.agent_y],
                    "reward": reward,
                    "finished": finished,
                    "metadata": result.metadata,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                trajectory.append(entry)

                input_list.append({
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": output_text,
                })

                print(f"  Step {step}: move {args.get('direction'):5s} -> "
                      f"pos=({env.agent_x},{env.agent_y}) "
                      f"explored={result.metadata.get('exploration_pct', 0)*100:.1f}% "
                      f"{'FINISHED' if finished else ''}")

                if finished:
                    break

    elapsed = time.time() - start_time

    summary = {
        "type": "summary",
        "task_id": task_spec["id"],
        "difficulty": task_spec["difficulty"],
        "grid_size": task_spec["grid_size"],
        "total_steps": step,
        "final_reward": reward,
        "termination": result.metadata.get("termination", "unknown"),
        "exploration_pct": result.metadata.get("exploration_pct", 0),
        "elapsed_seconds": round(elapsed, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    trajectory.append(summary)

    with open(trajectory_file, "a") as f:
        for entry in trajectory:
            f.write(json.dumps(entry) + "\n")

    await env.teardown()
    return summary


async def main():
    oai_client = AsyncOpenAI()
    model_name = "gpt-4.1-mini"

    trajectory_file = os.path.join(
        os.path.dirname(__file__), "trajectory.jsonl"
    )

    # Clear previous trajectory
    if os.path.exists(trajectory_file):
        os.remove(trajectory_file)

    tasks = MarsExplorerEnvironment.list_tasks("train")

    # Run 1 task per difficulty for testing
    test_tasks = []
    seen_difficulties = set()
    for t in tasks:
        if t["difficulty"] not in seen_difficulties:
            seen_difficulties.add(t["difficulty"])
            test_tasks.append(t)
        if len(test_tasks) == 3:
            break

    summaries = []
    for task_spec in test_tasks:
        print(f"\n{'='*60}")
        print(f"Task: {task_spec['id']} | {task_spec['difficulty']} "
              f"({task_spec['grid_size'][0]}x{task_spec['grid_size'][1]})")
        print(f"{'='*60}")

        summary = await run_task(oai_client, task_spec, model_name, trajectory_file)
        summaries.append(summary)

    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    for s in summaries:
        print(f"  {s['task_id']:25s} | {s['difficulty']:6s} | "
              f"steps={s['total_steps']:4d} | "
              f"explored={s['exploration_pct']*100:5.1f}% | "
              f"reward={s['final_reward']:.3f} | "
              f"{s['termination']} | "
              f"{s['elapsed_seconds']:.1f}s")

    print(f"\nTrajectory saved to: {trajectory_file}")


if __name__ == "__main__":
    asyncio.run(main())
