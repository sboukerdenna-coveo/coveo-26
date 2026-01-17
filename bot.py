import random
from typing import Optional

from game_message import *


class Bot:
    def __init__(self):
        print("Initializing exploration + constant production bot (with neutral handling)")
        self.visited: set[tuple[int, int]] = set()

        # Tuning knobs
        self.max_spores: int = 25
        self.base_spawn_biomass: int = 12
        self.spawn_biomass_cap: int = 256
        self.produce_every_tick: bool = True

        # Neutral handling knobs (lightweight on purpose)
        self.surrounded_threshold: int = 3   # "surrounded" if >= 3 adjacent neutral spores
        self.neutral_clear_radius: int = 8   # how far we consider neutrals as blockers when selecting targets

    # ---------- helpers ----------

    def _update_visited(self, game_message: TeamGameState) -> None:
        my_team: TeamInfo = game_message.world.teamInfos[game_message.yourTeamId]
        for sp in my_team.spores:
            self.visited.add((sp.position.x, sp.position.y))

    def _current_spawn_biomass(self, tick: int) -> int:
        multiplier = 2 ** (tick // 100)
        return min(self.base_spawn_biomass * multiplier, self.spawn_biomass_cap)

    def _adjacent_positions(self, x: int, y: int) -> list[tuple[int, int, Position]]:
        """
        Returns: (nx, ny, direction_vector) for 4-neighbors.
        direction_vector is what SporeMoveAction expects.
        """
        return [
            (x, y - 1, Position(x=0, y=-1)),
            (x, y + 1, Position(x=0, y=1)),
            (x - 1, y, Position(x=-1, y=0)),
            (x + 1, y, Position(x=1, y=0)),
        ]

    def _in_bounds(self, game_message: TeamGameState, x: int, y: int) -> bool:
        w, h = game_message.world.map.width, game_message.world.map.height
        return 0 <= x < w and 0 <= y < h

    def _neutral_spore_positions(self, game_message: TeamGameState) -> set[tuple[int, int]]:
        neutral_id = game_message.constants.neutralTeamId
        out: set[tuple[int, int]] = set()
        for sp in game_message.world.spores:
            if sp.teamId == neutral_id:
                out.add((sp.position.x, sp.position.y))
        return out

    def _is_surrounded_by_neutrals(
        self,
        game_message: TeamGameState,
        spore: Spore,
        neutral_pos: set[tuple[int, int]],
    ) -> list[tuple[Position, Position]]:
        """
        If adjacent tiles contain neutral spores, return list of (neutral_tile_pos, direction_vector).
        """
        x, y = spore.position.x, spore.position.y
        hits: list[tuple[Position, Position]] = []

        for nx, ny, dvec in self._adjacent_positions(x, y):
            if not self._in_bounds(game_message, nx, ny):
                continue
            if (nx, ny) in neutral_pos:
                hits.append((Position(x=nx, y=ny), dvec))

        return hits

    def _pick_unvisited_target(
        self,
        game_message: TeamGameState,
        start: Position,
        neutral_pos: set[tuple[int, int]],
        search_radius: int = 18,
        samples_outside: int = 80,
    ) -> Optional[Position]:
        """
        Prefer unvisited tiles, nutrient-biased.
        Avoid choosing a target tile that currently has a neutral spore on it.
        """
        world = game_message.world
        w, h = world.map.width, world.map.height
        nutrient = world.map.nutrientGrid

        best_pos: Optional[Position] = None
        best_score: float = float("-inf")

        sx, sy = start.x, start.y

        # Local window search
        x0 = max(0, sx - search_radius)
        x1 = min(w - 1, sx + search_radius)
        y0 = max(0, sy - search_radius)
        y1 = min(h - 1, sy + search_radius)

        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                if (x, y) in self.visited:
                    continue
                if (x, y) in neutral_pos:
                    continue  # can't take it until neutral is defeated
                dist = abs(x - sx) + abs(y - sy)
                if dist == 0:
                    continue

                score = (nutrient[y][x] * 3.0) - (dist * 1.0)
                if score > best_score:
                    best_score = score
                    best_pos = Position(x=x, y=y)

        if best_pos is not None:
            return best_pos

        # Fallback sampling
        for _ in range(samples_outside):
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            if (x, y) in self.visited or (x, y) in neutral_pos:
                continue
            dist = abs(x - sx) + abs(y - sy)
            if dist == 0:
                continue
            score = (nutrient[y][x] * 3.0) - (dist * 1.0)
            if score > best_score:
                best_score = score
                best_pos = Position(x=x, y=y)

        return best_pos

    def _nearest_neutral_in_radius(
        self,
        game_message: TeamGameState,
        start: Position,
        neutral_pos: set[tuple[int, int]],
        radius: int,
    ) -> Optional[Position]:
        """Light helper: find a nearby neutral spore tile (Manhattan distance)."""
        sx, sy = start.x, start.y
        best: Optional[Position] = None
        best_d = 10**9

        # cheap scan: neutrals are usually not massive
        for (nx, ny) in neutral_pos:
            d = abs(nx - sx) + abs(ny - sy)
            if d <= radius and d < best_d:
                best_d = d
                best = Position(x=nx, y=ny)
        return best

    # ---------- main ----------

    def get_next_move(self, game_message: TeamGameState) -> list[Action]:
        actions: list[Action] = []

        my_team: TeamInfo = game_message.world.teamInfos[game_message.yourTeamId]
        self._update_visited(game_message)

        neutral_pos = self._neutral_spore_positions(game_message)

        # 1) Ensure we have a spawner ASAP
        if len(my_team.spawners) == 0 and len(my_team.spores) > 0:
            actions.append(SporeCreateSpawnerAction(sporeId=my_team.spores[0].id))
            return actions

        # 2) Constantly produce spores (if we can afford it)
        if len(my_team.spawners) > 0 and self.produce_every_tick:
            spawn_biomass = self._current_spawn_biomass(game_message.tick)
            if len(my_team.spores) < self.max_spores and my_team.nutrients >= spawn_biomass:
                actions.append(
                    SpawnerProduceSporeAction(
                        spawnerId=my_team.spawners[0].id,
                        biomass=spawn_biomass,
                    )
                )

        # 3) Neutral “sacrifice” logic:
        # Find spores that are surrounded by neutrals; sacrifice the cheapest one to clear a blocker.
        surrounded_candidates: list[tuple[Spore, list[tuple[Position, Position]]]] = []
        for sp in my_team.spores:
            adjacent_neutrals = self._is_surrounded_by_neutrals(game_message, sp, neutral_pos)
            if len(adjacent_neutrals) >= self.surrounded_threshold:
                surrounded_candidates.append((sp, adjacent_neutrals))

        sacrificial_spore_id: Optional[str] = None
        sacrificial_move: Optional[SporeMoveAction] = None

        if surrounded_candidates:
            # pick lowest biomass surrounded spore (cheapest to lose)
            sp, adjacent_neutrals = min(surrounded_candidates, key=lambda t: t[0].biomass)

            # choose an adjacent neutral to ram (no need to be fancy)
            target_tile, dvec = adjacent_neutrals[0]
            sacrificial_spore_id = sp.id
            sacrificial_move = SporeMoveAction(sporeId=sp.id, direction=dvec)

        # 4) Movement: sacrificial spore clears neutrals, others expand to unvisited
        for sp in my_team.spores:
            if sacrificial_spore_id is not None and sp.id == sacrificial_spore_id:
                actions.append(sacrificial_move)  # type: ignore[arg-type]
                continue

            # If neutrals are nearby and blocking, sometimes it's better to clear them than wander.
            nearby_neutral = self._nearest_neutral_in_radius(
                game_message, sp.position, neutral_pos, radius=self.neutral_clear_radius
            )

            # Default target: unvisited expansion
            target = self._pick_unvisited_target(game_message, sp.position, neutral_pos)

            # Light preference: if we have no good unvisited target, or neutrals are close, go clear.
            if target is None and nearby_neutral is not None:
                target = nearby_neutral

            if target is None:
                w, h = game_message.world.map.width, game_message.world.map.height
                target = Position(x=random.randint(0, w - 1), y=random.randint(0, h - 1))

            actions.append(SporeMoveToAction(sporeId=sp.id, position=target))

        return actions
