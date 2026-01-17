import random
from typing import Optional

from game_message import *


class Bot:
    def __init__(self):
        print("Initializing exploration + constant production bot")
        self.visited: set[tuple[int, int]] = set()
        self.max_spores: int = 25          # cap to avoid overspending / overcrowding
        self.spawn_biomass: int = 12       # biomass for new spores (costs nutrients)
        self.produce_every_tick: bool = True

    # ---------- helpers ----------

    def _in_bounds(self, w: int, h: int, x: int, y: int) -> bool:
        return 0 <= x < w and 0 <= y < h

    def _update_visited(self, game_message: TeamGameState) -> None:
        """Mark current spore positions as visited (persistent across ticks)."""
        my_team: TeamInfo = game_message.world.teamInfos[game_message.yourTeamId]
        for sp in my_team.spores:
            self.visited.add((sp.position.x, sp.position.y))

    def _pick_unvisited_target(
        self,
        game_message: TeamGameState,
        start: Position,
        search_radius: int = 18,
        samples_outside: int = 80,
    ) -> Optional[Position]:
        """
        Prefer nearest unvisited tile, with a nutrient bias.
        We do a local scan (fast), then random sampling (fallback).
        """
        world = game_message.world
        w, h = world.map.width, world.map.height
        nutrient = world.map.nutrientGrid

        best_pos: Optional[Position] = None
        best_score: float = float("-inf")

        sx, sy = start.x, start.y

        # 1) Local window search (deterministic-ish, fast)
        x0 = max(0, sx - search_radius)
        x1 = min(w - 1, sx + search_radius)
        y0 = max(0, sy - search_radius)
        y1 = min(h - 1, sy + search_radius)

        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                if (x, y) in self.visited:
                    continue
                dist = abs(x - sx) + abs(y - sy)
                if dist == 0:
                    continue

                # Score = nutrient reward - distance penalty
                # tweak weights as you like
                score = (nutrient[y][x] * 3.0) - (dist * 1.0)

                if score > best_score:
                    best_score = score
                    best_pos = Position(x=x, y=y)

        if best_pos is not None:
            return best_pos

        # 2) Fallback: random sampling across the whole map
        for _ in range(samples_outside):
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            if (x, y) in self.visited:
                continue
            dist = abs(x - sx) + abs(y - sy)
            if dist == 0:
                continue
            score = (nutrient[y][x] * 3.0) - (dist * 1.0)
            if score > best_score:
                best_score = score
                best_pos = Position(x=x, y=y)

        return best_pos

    # ---------- main ----------

    def get_next_move(self, game_message: TeamGameState) -> list[Action]:
        actions: list[Action] = []

        my_team: TeamInfo = game_message.world.teamInfos[game_message.yourTeamId]
        self._update_visited(game_message)

        # 1) Ensure we have a spawner ASAP
        if len(my_team.spawners) == 0 and len(my_team.spores) > 0:
            actions.append(SporeCreateSpawnerAction(sporeId=my_team.spores[0].id))
            return actions  # priority action

        # 2) Constantly produce spores (if we can afford it)
        if len(my_team.spawners) > 0 and self.produce_every_tick:
            # keep producing until we hit cap or run out of nutrients
            # (engine might limit 1 action per spawner per tick; still safe to attempt one)
            if len(my_team.spores) < self.max_spores and my_team.nutrients >= self.spawn_biomass:
                actions.append(
                    SpawnerProduceSporeAction(
                        spawnerId=my_team.spawners[0].id,
                        biomass=self.spawn_biomass,
                    )
                )

        # 3) Exploration: move each spore toward an unvisited tile
        for sp in my_team.spores:
            target = self._pick_unvisited_target(game_message, sp.position)
            if target is None:
                # everything seems visited -> drift toward best nutrient tile randomly
                w, h = game_message.world.map.width, game_message.world.map.height
                target = Position(x=random.randint(0, w - 1), y=random.randint(0, h - 1))

            actions.append(SporeMoveToAction(sporeId=sp.id, position=target))

        return actions
