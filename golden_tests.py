"""Deterministic tests for MarsExplorer environment."""

import asyncio
import pytest

from marsexplorer import MarsExplorerEnvironment, DIFFICULTY_CONFIGS, DIFFICULTY_TASK_COUNTS


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def make_env(seed=0, difficulty="small", **overrides):
    config = DIFFICULTY_CONFIGS[difficulty].copy()
    config.update(overrides)
    task_spec = {
        "id": f"test_{difficulty}_{seed}",
        "seed": seed,
        "grid_size": config["grid_size"],
        "num_obstacles": config["num_obstacles"],
        "obstacle_size": config["obstacle_size"],
        "max_steps": config["max_steps"],
        "lidar_range": config["lidar_range"],
        "lidar_channels": config["lidar_channels"],
        "difficulty": difficulty,
        "movement_cost": config["movement_cost"],
        "bonus_reward": config["bonus_reward"],
        "collision_reward": config["collision_reward"],
        "out_of_bounds_reward": config["out_of_bounds_reward"],
    }
    return MarsExplorerEnvironment(task_spec)


async def setup_env(seed=0, difficulty="small", **overrides):
    env = make_env(seed, difficulty, **overrides)
    await env.setup()
    return env


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_splits():
    splits = MarsExplorerEnvironment.list_splits()
    assert splits == ["train"]


@pytest.mark.asyncio
async def test_list_tasks_count():
    tasks = MarsExplorerEnvironment.list_tasks("train")
    assert len(tasks) == 1000


@pytest.mark.asyncio
async def test_list_tasks_difficulty_distribution():
    tasks = MarsExplorerEnvironment.list_tasks("train")
    counts = {}
    for t in tasks:
        d = t["difficulty"]
        counts[d] = counts.get(d, 0) + 1
    assert counts["small"] == 334
    assert counts["medium"] == 333
    assert counts["large"] == 333


@pytest.mark.asyncio
async def test_list_tasks_invalid_split():
    with pytest.raises(ValueError):
        MarsExplorerEnvironment.list_tasks("test")


@pytest.mark.asyncio
async def test_map_reproducibility():
    env1 = await setup_env(seed=42, difficulty="medium")
    env2 = await setup_env(seed=42, difficulty="medium")
    assert env1.ground_truth == env2.ground_truth


@pytest.mark.asyncio
async def test_different_seeds_different_maps():
    env1 = await setup_env(seed=0, difficulty="medium")
    env2 = await setup_env(seed=1, difficulty="medium")
    assert env1.ground_truth != env2.ground_truth


@pytest.mark.asyncio
async def test_starting_position_free():
    for seed in range(20):
        for difficulty in ["small", "medium", "large"]:
            env = await setup_env(seed=seed, difficulty=difficulty)
            assert env.ground_truth[0][0] == 0, f"Start blocked: seed={seed}, diff={difficulty}"
            assert env.agent_x == 0
            assert env.agent_y == 0


@pytest.mark.asyncio
async def test_initial_lidar_reveals_cells():
    env = await setup_env(seed=0, difficulty="small")
    explored_count = sum(
        1 for y in range(env.grid_h) for x in range(env.grid_w)
        if env.explored[y][x]
    )
    assert explored_count > 0, "LIDAR should reveal at least the starting cell"


@pytest.mark.asyncio
async def test_get_prompt():
    env = await setup_env(seed=0, difficulty="small")
    prompt = await env.get_prompt()
    assert len(prompt) == 1
    text = prompt[0].text
    assert "move" in text.lower()
    assert "@" in text
    assert "?" in text


@pytest.mark.asyncio
async def test_valid_move_right():
    env = await setup_env(seed=0, difficulty="small")
    from marsexplorer import MoveParams
    result = await env.move(MoveParams(direction="right"))
    assert not result.finished
    assert env.agent_x == 1
    assert env.agent_y == 0
    assert env.step_count == 1
    # Dense reward: new cells - movement cost
    assert result.metadata["new_cells"] >= 0
    assert result.reward == result.metadata["new_cells"] - 0.2


@pytest.mark.asyncio
async def test_valid_move_down():
    env = await setup_env(seed=0, difficulty="small")
    from marsexplorer import MoveParams
    result = await env.move(MoveParams(direction="down"))
    assert not result.finished
    assert env.agent_x == 0
    assert env.agent_y == 1
    assert env.step_count == 1


@pytest.mark.asyncio
async def test_out_of_bounds_left():
    env = await setup_env(seed=0, difficulty="small")
    from marsexplorer import MoveParams
    result = await env.move(MoveParams(direction="left"))
    assert result.finished
    assert result.metadata["termination"] == "out_of_bounds"
    assert result.reward == -400.0


@pytest.mark.asyncio
async def test_out_of_bounds_up():
    env = await setup_env(seed=0, difficulty="small")
    from marsexplorer import MoveParams
    result = await env.move(MoveParams(direction="up"))
    assert result.finished
    assert result.metadata["termination"] == "out_of_bounds"


@pytest.mark.asyncio
async def test_invalid_direction():
    env = await setup_env(seed=0, difficulty="small")
    from marsexplorer import MoveParams
    result = await env.move(MoveParams(direction="diagonal"))
    assert not result.finished
    assert result.metadata.get("error") == "invalid_direction"


@pytest.mark.asyncio
async def test_move_after_game_over():
    env = await setup_env(seed=0, difficulty="small")
    from marsexplorer import MoveParams
    # End the game by going OOB
    await env.move(MoveParams(direction="left"))
    # Try to move again
    result = await env.move(MoveParams(direction="right"))
    assert result.finished
    assert result.metadata.get("error") == "game_finished"


@pytest.mark.asyncio
async def test_collision_detection():
    """Seed 0 medium has an obstacle at (1,4). Walk down 4 then right into it."""
    env = await setup_env(seed=0, difficulty="medium")
    from marsexplorer import MoveParams

    # Verify obstacle is where we expect
    assert env.ground_truth[4][1] == 1, "Expected obstacle at (1,4) for seed=0 medium"

    # Walk down 4 steps to (0, 4)
    for _ in range(4):
        result = await env.move(MoveParams(direction="down"))
        assert not result.finished

    assert env.agent_x == 0
    assert env.agent_y == 4

    # Move right into obstacle at (1, 4)
    result = await env.move(MoveParams(direction="right"))
    assert result.finished
    assert result.metadata["termination"] == "collision"
    assert result.reward == -400.0


@pytest.mark.asyncio
async def test_max_steps_termination():
    """Use a tiny max_steps to force termination."""
    env = await setup_env(seed=0, difficulty="small", max_steps=3)
    from marsexplorer import MoveParams

    for i in range(3):
        result = await env.move(MoveParams(direction="right"))
        if result.finished:
            # Might have hit something or explored 95%
            return

    assert result.finished
    assert result.metadata["termination"] == "max_steps"


@pytest.mark.asyncio
async def test_exploration_percentage_increases():
    env = await setup_env(seed=0, difficulty="small")
    from marsexplorer import MoveParams

    initial_explored, total = env._compute_exploration_stats()
    initial_pct = initial_explored / total

    # Move right a few times
    for _ in range(3):
        result = await env.move(MoveParams(direction="right"))
        if result.finished:
            break

    final_explored, total = env._compute_exploration_stats()
    final_pct = final_explored / total

    assert final_pct >= initial_pct, "Exploration should not decrease"


@pytest.mark.asyncio
async def test_render_map_contains_agent():
    env = await setup_env(seed=0, difficulty="small")
    rendered = env._render_map()
    assert "@" in rendered


@pytest.mark.asyncio
async def test_all_difficulties_produce_valid_maps():
    for difficulty in ["small", "medium", "large"]:
        env = await setup_env(seed=0, difficulty=difficulty)
        h, w = DIFFICULTY_CONFIGS[difficulty]["grid_size"]
        assert env.grid_h == h
        assert env.grid_w == w
        assert len(env.ground_truth) == h
        assert len(env.ground_truth[0]) == w


@pytest.mark.asyncio
async def test_dense_step_reward():
    """Each step reward = new_explored_cells - movement_cost."""
    env = await setup_env(seed=0, difficulty="small")
    from marsexplorer import MoveParams

    prev_explored = env.prev_explored_count

    result = await env.move(MoveParams(direction="right"))
    assert not result.finished

    new_explored, _ = env._compute_exploration_stats()
    new_cells = new_explored - prev_explored
    expected_reward = new_cells - env.config.movement_cost
    assert result.reward == expected_reward
    assert result.metadata["new_cells"] == new_cells
    assert result.metadata["step_reward"] == expected_reward


@pytest.mark.asyncio
async def test_step_reward_when_no_new_cells():
    """Moving to an already-explored area gives reward = -movement_cost."""
    env = await setup_env(seed=0, difficulty="small")
    from marsexplorer import MoveParams

    # Move right then left -- the left move reveals no new cells
    await env.move(MoveParams(direction="right"))
    result = await env.move(MoveParams(direction="left"))
    assert not result.finished
    # Should be -0.2 (no new cells, only movement cost)
    assert result.reward == -env.config.movement_cost


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
