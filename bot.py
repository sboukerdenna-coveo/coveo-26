import heapq
import random
from typing import Optional

from game_message import *


class Bot:
    def __init__(self):
        print("Initializing exploration + constant production bot (with neutrals + A*)")
        self.visited: set[tuple[int, int]] = set()

        # Tuning knobs
        self.max_spores: int = 25
        self.base_spawn_biomass: int = 4
        self.spawn_biomass_cap: int = 256
        self.produce_every_tick: bool = True

        # Neutral handling knobs (lightweight)
        self.surrounded_threshold: int = 3
        self.neutral_clear_radius: int = 8

        # Attack spawners if near
        self.attack_spawner_radius: int = 12

        # Panic-spawner settings
        self.enemy_threat_radius: int = 3
        self.panic_spawner_cooldown: int = 30  # don't do it every tick
        self._last_panic_spawner_tick: int = -10**9

        # Tick 0 spawner burst
        self._did_initial_spawner_burst: bool = False

        # Pathfinding knobs
        self.astar_max_expansions: int = 4000  # safety cap so we don't explode CPU

    # ---------- basic helpers ----------

    def _update_visited(self, game_message: TeamGameState) -> None:
        my_team: TeamInfo = game_message.world.teamInfos[game_message.yourTeamId]
        for sp in my_team.spores:
            self.visited.add((sp.position.x, sp.position.y))

    def _current_spawn_biomass(self, tick: int) -> int:
        multiplier = 2 ** (tick // 150)
        return min(self.base_spawn_biomass * multiplier, self.spawn_biomass_cap)

    def _in_bounds(self, game_message: TeamGameState, x: int, y: int) -> bool:
        w, h = game_message.world.map.width, game_message.world.map.height
        return 0 <= x < w and 0 <= y < h

    def _adjacent_positions(self, x: int, y: int) -> list[tuple[int, int, Position]]:
        # 4-neighbors with direction vectors for SporeMoveAction
        return [
            (x, y - 1, Position(x=0, y=-1)),
            (x, y + 1, Position(x=0, y=1)),
            (x - 1, y, Position(x=-1, y=0)),
            (x + 1, y, Position(x=1, y=0)),
        ]

    def _neutral_spore_positions(self, game_message: TeamGameState) -> set[tuple[int, int]]:
        neutral_id = game_message.constants.neutralTeamId
        return {
            (sp.position.x, sp.position.y)
            for sp in game_message.world.spores
            if sp.teamId == neutral_id
        }

    def _neutral_spore_biomass_by_pos(self, game_message: TeamGameState) -> dict[tuple[int, int], int]:
        """Map (x,y) -> neutral biomass (used to pick cheapest adjacent neutral)."""
        neutral_id = game_message.constants.neutralTeamId
        out: dict[tuple[int, int], int] = {}
        for sp in game_message.world.spores:
            if sp.teamId == neutral_id:
                out[(sp.position.x, sp.position.y)] = sp.biomass
        return out

    def _enemy_spore_positions(self, game_message: TeamGameState) -> set[tuple[int, int]]:
        # "Enemy" here means any non-neutral, non-us spore
        my_id = game_message.yourTeamId
        neutral_id = game_message.constants.neutralTeamId
        out: set[tuple[int, int]] = set()
        for sp in game_message.world.spores:
            if sp.teamId != my_id and sp.teamId != neutral_id:
                out.add((sp.position.x, sp.position.y))
        return out

    def _enemy_spawner_positions(self, game_message: TeamGameState) -> set[tuple[int, int]]:
        """All enemy spawner tile positions."""
        my_id = game_message.yourTeamId
        neutral_id = game_message.constants.neutralTeamId
        out: set[tuple[int, int]] = set()
        for spw in game_message.world.spawners:
            if spw.teamId != my_id and spw.teamId != neutral_id:
                out.add((spw.position.x, spw.position.y))
        return out

    def _manhattan(self, ax: int, ay: int, bx: int, by: int) -> int:
        return abs(ax - bx) + abs(ay - by)

    # ---------- threat + panic spawner ----------

    def _enemy_nearby(self, game_message: TeamGameState, enemy_pos: set[tuple[int, int]]) -> bool:
        """True if any enemy spore is within enemy_threat_radius of any of our spores."""
        my_team: TeamInfo = game_message.world.teamInfos[game_message.yourTeamId]
        if not enemy_pos or not my_team.spores:
            return False

        r = self.enemy_threat_radius
        for sp in my_team.spores:
            sx, sy = sp.position.x, sp.position.y
            for ex, ey in enemy_pos:
                if self._manhattan(sx, sy, ex, ey) <= r:
                    return True
        return False

    def _pick_safest_spore_for_new_spawner(
        self,
        game_message: TeamGameState,
        enemy_pos: set[tuple[int, int]],
    ) -> Optional[Spore]:
        """
        Choose the spore that is farthest from the nearest enemy spore.
        That approximates "somewhere else" (safer backline).
        """
        my_team: TeamInfo = game_message.world.teamInfos[game_message.yourTeamId]
        if not my_team.spores:
            return None
        if not enemy_pos:
            return max(my_team.spores, key=lambda s: (s.position.x, s.position.y))

        def min_dist_to_enemy(sp: Spore) -> int:
            sx, sy = sp.position.x, sp.position.y
            return min(self._manhattan(sx, sy, ex, ey) for ex, ey in enemy_pos)

        return max(my_team.spores, key=min_dist_to_enemy)

    # ---------- strategy target selection ----------

    def _is_surrounded_by_neutrals(
        self,
        game_message: TeamGameState,
        spore: Spore,
        neutral_pos: set[tuple[int, int]],
    ) -> list[tuple[Position, Position]]:
        """Return adjacent neutral tiles and the direction to move into them."""
        x, y = spore.position.x, spore.position.y
        hits: list[tuple[Position, Position]] = []
        for nx, ny, dvec in self._adjacent_positions(x, y):
            if not self._in_bounds(game_message, nx, ny):
                continue
            if (nx, ny) in neutral_pos:
                hits.append((Position(x=nx, y=ny), dvec))
        return hits

    def _nearest_neutral_in_radius(
        self,
        start: Position,
        neutral_pos: set[tuple[int, int]],
        radius: int,
    ) -> Optional[Position]:
        sx, sy = start.x, start.y
        best: Optional[Position] = None
        best_d = 10**9
        for (nx, ny) in neutral_pos:
            d = abs(nx - sx) + abs(ny - sy)
            if d <= radius and d < best_d:
                best_d = d
                best = Position(x=nx, y=ny)
        return best

    def _nearest_enemy_spawner_in_radius(
        self,
        start: Position,
        enemy_spawner_pos: set[tuple[int, int]],
        radius: int,
    ) -> Optional[Position]:
        sx, sy = start.x, start.y
        best: Optional[Position] = None
        best_d = 10**9
        for (ex, ey) in enemy_spawner_pos:
            d = abs(ex - sx) + abs(ey - sy)
            if d <= radius and d < best_d:
                best_d = d
                best = Position(x=ex, y=ey)
        return best

    def _pick_unvisited_target(
        self,
        game_message: TeamGameState,
        start: Position,
        blocked: set[tuple[int, int]],
        search_radius: int = 25,
        samples_outside: int = 80,
    ) -> Optional[Position]:
        """
        Prefer unvisited tiles, nutrient-biased.
        Avoid blocked tiles (neutrals/enemies, enemy spawners).
        """
        world = game_message.world
        w, h = world.map.width, world.map.height
        nutrient = world.map.nutrientGrid

        sx, sy = start.x, start.y
        best_pos: Optional[Position] = None
        best_score: float = float("-inf")

        x0 = max(0, sx - search_radius)
        x1 = min(w - 1, sx + search_radius)
        y0 = max(0, sy - search_radius)
        y1 = min(h - 1, sy + search_radius)

        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                if (x, y) in self.visited:
                    continue
                if (x, y) in blocked:
                    continue
                dist = abs(x - sx) + abs(y - sy)
                if dist == 0:
                    continue

                score = (nutrient[y][x] * 3.0) - (dist * 1.0)
                if score > best_score:
                    best_score = score
                    best_pos = Position(x=x, y=y)

        if best_pos is not None:
            return best_pos

        for _ in range(samples_outside):
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            if (x, y) in self.visited or (x, y) in blocked:
                continue
            dist = abs(x - sx) + abs(y - sy)
            if dist == 0:
                continue
            score = (nutrient[y][x] * 3.0) - (dist * 1.0)
            if score > best_score:
                best_score = score
                best_pos = Position(x=x, y=y)

        return best_pos

    def _choose_sacrifice_target(
        self,
        game_message: TeamGameState,
        my_spores: list[Spore],
        neutral_pos: set[tuple[int, int]],
        neutral_biomass: dict[tuple[int, int], int],
    ) -> tuple[Optional[str], Optional[SporeMoveAction]]:
        """
        Sacrifice mode (when surrounded by neutrals): pick lowest-biomass spore,
        ram cheapest adjacent neutral.
        """
        candidates: list[tuple[Spore, tuple[Position, Position]]] = []

        for sp in my_spores:
            adj = self._is_surrounded_by_neutrals(game_message, sp, neutral_pos)
            if len(adj) < self.surrounded_threshold:
                continue

            best_tile, best_dir = adj[0]
            best_cost = neutral_biomass.get((best_tile.x, best_tile.y), 10**9)
            for tile_pos, dvec in adj[1:]:
                cost = neutral_biomass.get((tile_pos.x, tile_pos.y), 10**9)
                if cost < best_cost:
                    best_cost = cost
                    best_tile, best_dir = tile_pos, dvec

            candidates.append((sp, (best_tile, best_dir)))

        if not candidates:
            return None, None

        sp, (_tile, dvec) = min(candidates, key=lambda t: t[0].biomass)
        return sp.id, SporeMoveAction(sporeId=sp.id, direction=dvec)

    # ---------- A* pathfinding ----------

    def _heuristic(self, ax: int, ay: int, bx: int, by: int) -> int:
        # Manhattan distance (perfect for 4-neighbor grid without diagonal moves)
        return abs(ax - bx) + abs(ay - by)

    def _astar_next_step(
        self,
        game_message: TeamGameState,
        start: Position,
        goal: Position,
        blocked: set[tuple[int, int]],
        allow_goal_even_if_blocked: bool = False,
    ) -> Optional[Position]:
        """
        Returns the direction vector (Position) for the next step from start toward goal using A*.
        If no path found, returns None.
        """
        sx, sy = start.x, start.y
        gx, gy = goal.x, goal.y

        if (sx, sy) == (gx, gy):
            return None

        if not self._in_bounds(game_message, gx, gy):
            return None

        if (gx, gy) in blocked and not allow_goal_even_if_blocked:
            return None

        open_heap: list[tuple[int, int, int, int]] = []  # (f, g, x, y)
        heapq.heappush(open_heap, (self._heuristic(sx, sy, gx, gy), 0, sx, sy))

        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        g_score: dict[tuple[int, int], int] = {(sx, sy): 0}

        expansions = 0

        while open_heap and expansions < self.astar_max_expansions:
            f, g, x, y = heapq.heappop(open_heap)

            if (x, y) == (gx, gy):
                cur = (gx, gy)
                prev = came_from.get(cur)
                while prev is not None and prev != (sx, sy):
                    cur = prev
                    prev = came_from.get(cur)

                nx, ny = cur
                dx = nx - sx
                dy = ny - sy
                return Position(x=dx, y=dy)

            expansions += 1

            for nx, ny, _dvec in self._adjacent_positions(x, y):
                if not self._in_bounds(game_message, nx, ny):
                    continue

                if (nx, ny) in blocked and not (allow_goal_even_if_blocked and (nx, ny) == (gx, gy)):
                    continue

                tentative_g = g + 1
                key = (nx, ny)

                if tentative_g < g_score.get(key, 10**9):
                    came_from[key] = (x, y)
                    g_score[key] = tentative_g
                    h = self._heuristic(nx, ny, gx, gy)
                    heapq.heappush(open_heap, (tentative_g + h, tentative_g, nx, ny))

        return None

    # ---------- main ----------

    def get_next_move(self, game_message: TeamGameState) -> list[Action]:
        actions: list[Action] = []

        my_team: TeamInfo = game_message.world.teamInfos[game_message.yourTeamId]
        self._update_visited(game_message)

        neutral_pos = self._neutral_spore_positions(game_message)
        neutral_biomass = self._neutral_spore_biomass_by_pos(game_message)
        enemy_pos = self._enemy_spore_positions(game_message)
        enemy_spawner_pos = self._enemy_spawner_positions(game_message)

        # Treat neutrals + enemies + enemy spawners as blocked by default
        blocked_for_path = set(neutral_pos) | set(enemy_pos) | set(enemy_spawner_pos)

        # Tick 0: put a spawner on every starting tile (every starting spore position)
        if game_message.tick == 0 and not self._did_initial_spawner_burst:
            self._did_initial_spawner_burst = True
            for sp in my_team.spores:
                actions.append(SporeCreateSpawnerAction(sporeId=sp.id))
            return actions

        # PANIC: enemy close -> create a new spawner somewhere safer (cooldown)
        enemy_threat = self._enemy_nearby(game_message, enemy_pos)
        if enemy_threat and (game_message.tick - self._last_panic_spawner_tick >= self.panic_spawner_cooldown):
            safest = self._pick_safest_spore_for_new_spawner(game_message, enemy_pos)
            if safest is not None and len(my_team.spawners) < 3:
                actions.append(SporeCreateSpawnerAction(sporeId=safest.id))
                self._last_panic_spawner_tick = game_message.tick
                return actions

        # Ensure we have a spawner ASAP (fallback)
        if len(my_team.spawners) == 0 and len(my_team.spores) > 0:
            actions.append(SporeCreateSpawnerAction(sporeId=my_team.spores[0].id))
            return actions

        # Constantly produce spores (if we can afford it)
        if len(my_team.spawners) > 0 and self.produce_every_tick:
            spawn_biomass = self._current_spawn_biomass(game_message.tick)
            if len(my_team.spores) < self.max_spores and my_team.nutrients >= spawn_biomass:
                actions.append(
                    SpawnerProduceSporeAction(
                        spawnerId=my_team.spawners[0].id,
                        biomass=spawn_biomass,
                    )
                )

        # ---------- MINIMAL CHANGE: only sacrifice if no unvisited targets exist ----------
        any_unvisited_target = False
        for sp in my_team.spores:
            if self._pick_unvisited_target(game_message, sp.position, blocked_for_path) is not None:
                any_unvisited_target = True
                break

        if any_unvisited_target:
            sacrificial_spore_id, sacrificial_move = None, None
        else:
            sacrificial_spore_id, sacrificial_move = self._choose_sacrifice_target(
                game_message=game_message,
                my_spores=my_team.spores,
                neutral_pos=neutral_pos,
                neutral_biomass=neutral_biomass,
            )
        # -------------------------------------------------------------------------------

        # Movement: attack enemy spawner if near, else expand
        for sp in my_team.spores:
            # Sacrifice move overrides pathfinding (but only when expansion is impossible)
            if sacrificial_spore_id is not None and sp.id == sacrificial_spore_id:
                actions.append(sacrificial_move)  # type: ignore[arg-type]
                continue

            # Attack nearby enemy spawner
            spawner_target = self._nearest_enemy_spawner_in_radius(
                sp.position, enemy_spawner_pos, self.attack_spawner_radius
            )

            if spawner_target is not None:
                target = spawner_target
                allow_goal_blocked = True  # spawner is in blocked_for_path; allow stepping onto it
            else:
                # Prefer exploration to unvisited tiles (avoid blocked tiles)
                target = self._pick_unvisited_target(game_message, sp.position, blocked_for_path)

                # If no unvisited target found, but neutrals are close, go clear a nearby neutral
                if target is None:
                    nearby_neutral = self._nearest_neutral_in_radius(
                        sp.position, neutral_pos, self.neutral_clear_radius
                    )
                    if nearby_neutral is not None:
                        target = nearby_neutral
                    else:
                        w, h = game_message.world.map.width, game_message.world.map.height
                        target = Position(x=random.randint(0, w - 1), y=random.randint(0, h - 1))

                # If target is a neutral tile, we allow goal to be "blocked" because we want to fight it.
                allow_goal_blocked = (target.x, target.y) in neutral_pos

            next_dir = self._astar_next_step(
                game_message=game_message,
                start=sp.position,
                goal=target,
                blocked=blocked_for_path,
                allow_goal_even_if_blocked=allow_goal_blocked,
            )

            if next_dir is None:
                actions.append(SporeMoveToAction(sporeId=sp.id, position=target))
            else:
                actions.append(SporeMoveAction(sporeId=sp.id, direction=next_dir))

        return actions
