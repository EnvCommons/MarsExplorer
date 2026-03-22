import math
import numpy as np
from typing import List
from pydantic import BaseModel

from openreward.environments import Environment, JSONObject, ToolOutput, tool, TextBlock


# ============================================================
# Task & Tool Parameter Models
# ============================================================


class TaskSpec(BaseModel):
    id: str
    seed: int
    grid_size: list[int]       # [height, width]
    num_obstacles: int
    obstacle_size: list[int]   # [min, max]
    max_steps: int
    lidar_range: int
    lidar_channels: int
    difficulty: str
    movement_cost: float
    bonus_reward: float
    collision_reward: float
    out_of_bounds_reward: float


class MoveParams(BaseModel, extra="forbid"):
    direction: str  # "up", "down", "left", "right"


# ============================================================
# Difficulty configurations
# ============================================================

DIFFICULTY_CONFIGS = {
    "small": {
        "grid_size": [11, 11],
        "num_obstacles": 5,
        "obstacle_size": [1, 2],
        "max_steps": 150,
        "lidar_range": 4,
        "lidar_channels": 32,
        "movement_cost": 0.2,
        "bonus_reward": 400.0,
        "collision_reward": -400.0,
        "out_of_bounds_reward": -400.0,
    },
    "medium": {
        "grid_size": [21, 21],
        "num_obstacles": 12,
        "obstacle_size": [2, 2],
        "max_steps": 400,
        "lidar_range": 6,
        "lidar_channels": 32,
        "movement_cost": 0.2,
        "bonus_reward": 400.0,
        "collision_reward": -400.0,
        "out_of_bounds_reward": -400.0,
    },
    "large": {
        "grid_size": [41, 41],
        "num_obstacles": 30,
        "obstacle_size": [2, 3],
        "max_steps": 1000,
        "lidar_range": 8,
        "lidar_channels": 32,
        "movement_cost": 0.2,
        "bonus_reward": 400.0,
        "collision_reward": -400.0,
        "out_of_bounds_reward": -400.0,
    },
}

DIFFICULTY_TASK_COUNTS = {
    "small": 334,
    "medium": 333,
    "large": 333,
}


# ============================================================
# Main Environment Class
# ============================================================


class MarsExplorerEnvironment(Environment):

    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        super().__init__(task_spec)
        self.config = TaskSpec.model_validate(task_spec)

        self.grid_h = 0
        self.grid_w = 0
        self.ground_truth: list[list[int]] = []
        self.explored: list[list[bool]] = []
        self.agent_x = 0
        self.agent_y = 0
        self.step_count = 0
        self.game_over = False
        self.termination_reason = ""
        self.prev_explored_count = 0

    async def setup(self):
        h, w = self.config.grid_size
        self.grid_h = h
        self.grid_w = w

        self.ground_truth = self._generate_map(
            seed=self.config.seed,
            grid_size=self.config.grid_size,
            num_obstacles=self.config.num_obstacles,
            obstacle_size=self.config.obstacle_size,
        )

        self.explored = [[False] * w for _ in range(h)]

        self.agent_x = 0
        self.agent_y = 0

        self._cast_and_reveal()

        self.prev_explored_count, _ = self._compute_exploration_stats()
        self.step_count = 0
        self.game_over = False
        self.termination_reason = ""

    async def teardown(self):
        pass

    # ----------------------------------------------------------
    # Map generation
    # ----------------------------------------------------------

    @staticmethod
    def _generate_map(
        seed: int,
        grid_size: list[int],
        num_obstacles: int,
        obstacle_size: list[int],
        margin: int = 1,
    ) -> list[list[int]]:
        rng = np.random.RandomState(seed)
        h, w = grid_size
        grid = [[0] * w for _ in range(h)]

        for _ in range(num_obstacles):
            ox = rng.randint(margin, max(margin + 1, w - margin))
            oy = rng.randint(margin, max(margin + 1, h - margin))
            oh = rng.randint(obstacle_size[0], obstacle_size[1] + 1)
            ow = rng.randint(obstacle_size[0], obstacle_size[1] + 1)
            for dy in range(oh):
                for dx in range(ow):
                    ny, nx = oy + dy, ox + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        grid[ny][nx] = 1

        # Ensure starting position and its immediate surroundings are free
        grid[0][0] = 0
        if w > 1:
            grid[0][1] = 0
        if h > 1:
            grid[1][0] = 0

        return grid

    # ----------------------------------------------------------
    # LIDAR
    # ----------------------------------------------------------

    def _cast_and_reveal(self):
        thetas = np.linspace(0, 2 * math.pi, self.config.lidar_channels, endpoint=False)
        resolution = self.config.lidar_range * 15

        for theta in thetas:
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            for step in range(resolution + 1):
                r = (step / resolution) * self.config.lidar_range
                cx = int(round(self.agent_x + r * cos_t))
                cy = int(round(self.agent_y + r * sin_t))

                if cx < 0 or cx >= self.grid_w or cy < 0 or cy >= self.grid_h:
                    break

                self.explored[cy][cx] = True

                if self.ground_truth[cy][cx] == 1:
                    break

    # ----------------------------------------------------------
    # Observation rendering
    # ----------------------------------------------------------

    def _compute_exploration_stats(self) -> tuple[int, int]:
        total_cells = self.grid_h * self.grid_w
        explored_count = sum(
            1 for y in range(self.grid_h) for x in range(self.grid_w)
            if self.explored[y][x]
        )
        return explored_count, total_cells

    def _render_map(self) -> str:
        lines = []

        if self.grid_w <= 50:
            tens_row = "   " + "".join(
                str(x // 10) if x >= 10 else " " for x in range(self.grid_w)
            )
            ones_row = "   " + "".join(str(x % 10) for x in range(self.grid_w))
            lines.append(tens_row)
            lines.append(ones_row)

        for y in range(self.grid_h):
            row_chars = []
            for x in range(self.grid_w):
                if x == self.agent_x and y == self.agent_y:
                    row_chars.append("@")
                elif not self.explored[y][x]:
                    row_chars.append("?")
                elif self.ground_truth[y][x] == 1:
                    row_chars.append("#")
                else:
                    row_chars.append(".")
            label = f"{y:2d} " if self.grid_w <= 50 else ""
            lines.append(label + "".join(row_chars))

        return "\n".join(lines)

    def _format_observation(self) -> str:
        explored_count, total_cells = self._compute_exploration_stats()
        pct = (explored_count / total_cells * 100) if total_cells > 0 else 0.0

        parts = [
            f"Step: {self.step_count}/{self.config.max_steps} | "
            f"Position: ({self.agent_x}, {self.agent_y}) | "
            f"Explored: {pct:.1f}% ({explored_count}/{total_cells})",
            "",
            self._render_map(),
            "",
            "Legend: @ = rover, # = obstacle, . = explored, ? = unexplored",
        ]
        return "\n".join(parts)

    # ----------------------------------------------------------
    # Tool
    # ----------------------------------------------------------

    DIRECTION_MAP = {
        "up": (0, -1),
        "down": (0, 1),
        "left": (-1, 0),
        "right": (1, 0),
    }

    @tool
    async def move(self, params: MoveParams) -> ToolOutput:
        """Move the rover one step in the given direction (up/down/left/right). Returns the updated map."""

        if self.game_over:
            return ToolOutput(
                blocks=[TextBlock(text="Mission is already over.")],
                metadata={"error": "game_finished"},
                reward=0.0,
                finished=True,
            )

        direction = params.direction.lower().strip()
        if direction not in self.DIRECTION_MAP:
            return ToolOutput(
                blocks=[TextBlock(
                    text=f"Invalid direction: '{direction}'. Use one of: up, down, left, right."
                )],
                metadata={"error": "invalid_direction"},
                reward=0.0,
                finished=False,
            )

        dx, dy = self.DIRECTION_MAP[direction]
        new_x = self.agent_x + dx
        new_y = self.agent_y + dy

        # Out of bounds
        if new_x < 0 or new_x >= self.grid_w or new_y < 0 or new_y >= self.grid_h:
            self.game_over = True
            self.termination_reason = "out_of_bounds"
            explored_count, total_cells = self._compute_exploration_stats()
            pct = explored_count / total_cells if total_cells > 0 else 0.0
            reward = self.config.out_of_bounds_reward

            return ToolOutput(
                blocks=[TextBlock(
                    text=f"MISSION FAILED: Rover moved out of bounds at ({new_x}, {new_y}).\n"
                         f"Final exploration: {pct * 100:.1f}% | Reward: {reward:.1f}"
                )],
                metadata={
                    "termination": "out_of_bounds",
                    "exploration_pct": round(pct, 4),
                    "steps_taken": self.step_count,
                    "reward": reward,
                },
                reward=reward,
                finished=True,
            )

        # Collision
        if self.ground_truth[new_y][new_x] == 1:
            self.game_over = True
            self.termination_reason = "collision"
            explored_count, total_cells = self._compute_exploration_stats()
            pct = explored_count / total_cells if total_cells > 0 else 0.0
            reward = self.config.collision_reward

            return ToolOutput(
                blocks=[TextBlock(
                    text=f"MISSION FAILED: Rover collided with obstacle at ({new_x}, {new_y}).\n"
                         f"Final exploration: {pct * 100:.1f}% | Reward: {reward:.1f}"
                )],
                metadata={
                    "termination": "collision",
                    "exploration_pct": round(pct, 4),
                    "steps_taken": self.step_count,
                    "reward": reward,
                },
                reward=reward,
                finished=True,
            )

        # Valid move
        self.agent_x = new_x
        self.agent_y = new_y
        self.step_count += 1
        self._cast_and_reveal()

        explored_count, total_cells = self._compute_exploration_stats()
        pct = explored_count / total_cells if total_cells > 0 else 0.0
        new_cells = explored_count - self.prev_explored_count
        self.prev_explored_count = explored_count

        # Per-step reward: new explored cells minus movement cost
        step_reward = new_cells - self.config.movement_cost

        # 95% explored -> success
        if pct >= 0.95:
            self.game_over = True
            self.termination_reason = "exploration_complete"
            reward = self.config.bonus_reward
            obs = self._format_observation()
            return ToolOutput(
                blocks=[TextBlock(
                    text=f"MISSION SUCCESS: {pct * 100:.1f}% of terrain explored!\n"
                         f"Bonus reward: {reward:.1f}\n\n{obs}"
                )],
                metadata={
                    "termination": "exploration_complete",
                    "exploration_pct": round(pct, 4),
                    "steps_taken": self.step_count,
                    "new_cells": new_cells,
                    "step_reward": step_reward,
                    "reward": reward,
                },
                reward=reward,
                finished=True,
            )

        # Max steps reached
        if self.step_count >= self.config.max_steps:
            self.game_over = True
            self.termination_reason = "max_steps"
            obs = self._format_observation()
            return ToolOutput(
                blocks=[TextBlock(
                    text=f"MISSION COMPLETE: Step limit reached.\n"
                         f"Final exploration: {pct * 100:.1f}% | Step reward: {step_reward:.1f}\n\n{obs}"
                )],
                metadata={
                    "termination": "max_steps",
                    "exploration_pct": round(pct, 4),
                    "steps_taken": self.step_count,
                    "new_cells": new_cells,
                    "step_reward": step_reward,
                    "reward": step_reward,
                },
                reward=step_reward,
                finished=True,
            )

        # Normal step -- dense reward
        obs = self._format_observation()
        return ToolOutput(
            blocks=[TextBlock(text=obs)],
            metadata={
                "exploration_pct": round(pct, 4),
                "position": [self.agent_x, self.agent_y],
                "step": self.step_count,
                "new_cells": new_cells,
                "step_reward": step_reward,
            },
            reward=step_reward,
            finished=False,
        )

    # ----------------------------------------------------------
    # Prompt
    # ----------------------------------------------------------

    async def get_prompt(self) -> List[TextBlock]:
        obs = self._format_observation()
        prompt = (
            f"You are controlling a Mars rover exploring unknown terrain on a "
            f"{self.grid_w}x{self.grid_h} grid.\n"
            f"\n"
            f"OBJECTIVE: Explore as much of the map as possible within {self.config.max_steps} steps.\n"
            f"\n"
            f"MECHANICS:\n"
            f"- Use the move tool with direction \"up\", \"down\", \"left\", or \"right\" "
            f"to move one cell at a time\n"
            f"- Your rover has a LIDAR sensor (range: {self.config.lidar_range}) that "
            f"automatically reveals nearby terrain after each move\n"
            f"- Moving into an obstacle (#) or out of bounds ends the mission immediately "
            f"with a penalty\n"
            f"- The mission succeeds when 95% or more of the terrain is explored\n"
            f"- The mission ends when you run out of steps\n"
            f"\n"
            f"MAP SYMBOLS:\n"
            f"- @ = your rover\n"
            f"- # = obstacle (revealed by LIDAR)\n"
            f"- . = explored free terrain\n"
            f"- ? = unexplored terrain\n"
            f"\n"
            f"STRATEGY TIPS:\n"
            f"- Plan efficient routes to cover unexplored areas (?)\n"
            f"- Avoid obstacles (#) -- collision ends the mission\n"
            f"- Stay within grid boundaries\n"
            f"- Prioritize exploring large unexplored regions\n"
            f"\n"
            f"CURRENT STATE:\n"
            f"{obs}\n"
            f"\n"
            f"Begin exploring. Move systematically to maximize terrain coverage."
        )
        return [TextBlock(text=prompt)]

    # ----------------------------------------------------------
    # Task management
    # ----------------------------------------------------------

    @classmethod
    def list_splits(cls) -> list[str]:
        return ["train"]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        if split != "train":
            raise ValueError(f"Unknown split: {split}. Only 'train' is available.")

        tasks = []
        for difficulty, config in DIFFICULTY_CONFIGS.items():
            count = DIFFICULTY_TASK_COUNTS[difficulty]
            for i in range(count):
                task_id = f"mars_{difficulty}_{i:04d}"
                tasks.append({
                    "id": task_id,
                    "seed": i,
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
                })

        return tasks
