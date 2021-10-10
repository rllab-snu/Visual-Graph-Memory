
from typing import Any, Dict, List, Optional, Type, Union
import numpy as np

from habitat.config import Config
from habitat.core.dataset import Dataset, Episode
from habitat.core.embodied_task import Measure
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.simulator import (
    AgentState,
    Simulator,
)
from habitat.core.utils import not_none_validator, try_cv2_import
from habitat.tasks.utils import (
    cartesian_to_polar
)
import math
import quaternion as q
from habitat.utils.visualizations import fog_of_war, maps
from habitat.tasks.nav.nav import TopDownMap
import os
cv2 = try_cv2_import()

# consider up to 5 agents
AGENTS = {}
LAST_INDEX = 10
COMMON_NODE = LAST_INDEX
COMMON_EDGE = LAST_INDEX + 1
CURR_NODE = LAST_INDEX + 2
LAST_INDEX += 3

MAP_THICKNESS_SCALAR: int = 1250

COORDINATE_MIN = -62.3241 - 1e-6
COORDINATE_MAX = 90.0399 + 1e-6
import torch
import copy

def to_grid(realworld_x, realworld_y, coordinate_min, coordinate_max, grid_resolution):
    grid_size = (
        (coordinate_max - coordinate_min) / grid_resolution[0],
        (coordinate_max - coordinate_min) / grid_resolution[1],
    )
    grid_x = int((coordinate_max - realworld_x) / grid_size[0])
    grid_y = int((realworld_y - coordinate_min) / grid_size[1])
    return grid_x, grid_y


def from_grid(grid_x, grid_y, coordinate_min, coordinate_max, grid_resolution):

    grid_size = (
        (coordinate_max - coordinate_min) / grid_resolution[0],
        (coordinate_max - coordinate_min) / grid_resolution[1],
    )
    realworld_x = coordinate_max - grid_x * grid_size[0]
    realworld_y = coordinate_min + grid_y * grid_size[1]
    return realworld_x, realworld_y


def get_topdown_map(
    sim: Simulator,
    map_resolution: (1250, 1250),
    num_samples: int = 20000,
    draw_border: bool = True,
    save_img: bool=True,
    draw_new_map: bool=False,
    loose_check: bool=False,
    height_th: float=0.1
) -> np.ndarray:
    r"""Return a top-down occupancy map for a sim. Note, this only returns valid
    values for whatever floor the agent is currently ogn.

    Args:
        sim: The simulator.
        map_resolution: The resolution of map which will be computed and
            returned.
        num_samples: The number of random navigable points which will be
            initially
            sampled. For large environments it may need to be increased.
        draw_border: Whether to outline the border of the occupied spaces.

    Returns:
        Image containing 0 if occupied, 1 if unoccupied, and 2 if border (if
        the flag is set).
    """
    top_down_map = np.zeros(map_resolution, dtype=np.uint8)
    border_padding = 0

    start_position = sim.get_agent_state().position
    start_height = start_position[1]
    scene_name = sim.habitat_config.SCENE.split('/')[-1][:-4]
    if map_resolution[0] == 1250:
        map_name = 'explore_map/%s_%.2f.png' % (scene_name, start_height)
    else:
        map_name = 'explore_map/%s_%.2f_res%d.png' % (scene_name, start_height, map_resolution[0])
    if os.path.exists(map_name) and not draw_new_map and map_resolution[0] == 1250:
        top_down_map = cv2.imread(map_name, cv2.IMREAD_GRAYSCALE)
    else:
        # Use sampling to find the extrema points that might be navigable.
        range_x = (map_resolution[0], 0)
        range_y = (map_resolution[1], 0)
        for _ in range(num_samples):
            point = sim.sample_navigable_point()
            # Check if on same level as original
            if np.abs(point[1] - start_height) > height_th: continue
            g_x, g_y = to_grid(
                point[0], point[2], COORDINATE_MIN, COORDINATE_MAX, map_resolution
            )
            top_down_map[g_x, g_y] = maps.MAP_VALID_POINT
            range_x = (min(range_x[0], g_x), max(range_x[1], g_x))
            range_y = (min(range_y[0], g_y), max(range_y[1], g_y))

        padding = int(np.ceil(map_resolution[0] / 125))
        range_x = (
            max(range_x[0] - padding, 0),
            min(range_x[-1] + padding + 1, top_down_map.shape[0]),
        )
        range_y = (
            max(range_y[0] - padding, 0),
            min(range_y[-1] + padding + 1, top_down_map.shape[1]),
        )

        if not loose_check:
            # # Search over grid for valid points.
            s_x, s_y = to_grid(
                    start_position[0], start_position[2], COORDINATE_MIN, COORDINATE_MAX, map_resolution
                )
            NEIGHBOR = np.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]], dtype=np.int32)
            pixel_distance = 0
            queue = [np.array([s_x,s_y, pixel_distance])]
            distance_map = np.ones_like(top_down_map) * -1
            while True:
                if len(queue) == 0: break
                pix_x, pix_y, distance = queue[0]
                del queue[0]
                if pix_x < range_x[0] or pix_x > range_x[1]: continue
                if pix_y < range_y[0] or pix_y > range_y[1] : continue
                if distance_map[pix_x, pix_y] != -1 : continue
                realworld_x, realworld_y = from_grid(
                    pix_x, pix_y, COORDINATE_MIN, COORDINATE_MAX, map_resolution
                )
                valid_point = sim.is_navigable(
                    [realworld_x, start_height, realworld_y]
                )

                top_down_map[pix_x, pix_y] = (
                    maps.MAP_VALID_POINT if valid_point else maps.MAP_INVALID_POINT
                )
                if valid_point:
                    distance_map[pix_x, pix_y] = distance
                    neighbor = np.array([pix_x, pix_y], dtype=np.int32) + NEIGHBOR
                    queue.extend(list(np.concatenate([neighbor, np.ones([len(neighbor),1],dtype=np.int32)*(distance+1)],1)))
            if save_img and top_down_map.sum() > 1000:
                cv2.imwrite(map_name, top_down_map)
        else:
            for xx in range(range_x[0], range_x[1]):
                for yy in range(range_y[0], range_y[1]):
                    valid_point = False
                    for dxx in range(-2,3):
                        for dyy in range(-2,3):
                            realworld_x, realworld_y = from_grid(
                                xx+dxx, yy+dyy, COORDINATE_MIN, COORDINATE_MAX, map_resolution
                            )
                            #sim.is_navigable(np.array([-5.411907, 0.15825206, -2.0683963]))
                            vvalid_point = sim.is_navigable(
                                [realworld_x, start_height, realworld_y]
                            )
                            if vvalid_point:
                                valid_point = True
                    valid_point = valid_point or top_down_map[xx, yy]
                    top_down_map[xx, yy] = (
                        maps.MAP_VALID_POINT if valid_point else maps.MAP_INVALID_POINT
                    )
            if save_img and top_down_map.sum() > 1000:
                cv2.imwrite(map_name, top_down_map)

    if draw_border:
        # Recompute range in case padding added any more values.
        range_x = np.where(np.any(top_down_map, axis=1))[0]
        range_y = np.where(np.any(top_down_map, axis=0))[0]
        if len(range_x) > 0 and len(range_y) > 0:
            range_x = (
                max(range_x[0] - border_padding, 0),
                min(range_x[-1] + border_padding + 1, top_down_map.shape[0]),
            )
            range_y = (
                max(range_y[0] - border_padding, 0),
                min(range_y[-1] + border_padding + 1, top_down_map.shape[1]),
            )

            maps._outline_border(
                top_down_map[range_x[0] : range_x[1], range_y[0] : range_y[1]]
            )
    return top_down_map

@registry.register_measure(name='TopDownGraphMap')
class TopDownGraphMap(Measure):
    r"""Top Down Map measure
    """
    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        maps.TOP_DOWN_MAP_COLORS[COMMON_NODE] = [119,91,138]
        maps.TOP_DOWN_MAP_COLORS[COMMON_EDGE] = [189,164,204]
        maps.TOP_DOWN_MAP_COLORS[CURR_NODE] = [94, 66, 118]

        maps.TOP_DOWN_MAP_COLORS[LAST_INDEX:] = cv2.applyColorMap(
            np.arange(256-LAST_INDEX, dtype=np.uint8), cv2.COLORMAP_JET
        ).squeeze(1)[:, ::-1]

        self._sim = sim
        self._config = config
        self._grid_delta = config.MAP_PADDING
        self._step_count = None
        self._map_resolution = (config.MAP_RESOLUTION, config.MAP_RESOLUTION)
        self._num_samples = config.NUM_TOPDOWN_MAP_SAMPLE_POINTS
        self._ind_x_min = None
        self._ind_x_max = None
        self._ind_y_min = None
        self._ind_y_max = None
        self._previous_xy_location = None
        self._coordinate_min = COORDINATE_MIN
        self._coordinate_max = COORDINATE_MAX
        self._top_down_map = None
        self._shortest_path_points = None
        self._cell_scale = (
            self._coordinate_max - self._coordinate_min
        ) / self._map_resolution[0]
        self.line_thickness = int(
            np.round(self._map_resolution[0] * 2 / MAP_THICKNESS_SCALAR)
        )
        self.point_padding = 2 * int(
            np.ceil(self._map_resolution[0] / MAP_THICKNESS_SCALAR)
        )
        self._previous_scene = None
        self._previous_position = None#self._sim.config.SCENE.split('/')[-2]
        self.delta = 12
        self.milli_delta = 60
        self.delta_angs = [(2*np.pi*i/self.milli_delta) for i in range(self.milli_delta)]
        self.delta_angs = self.delta_angs[30:] + self.delta_angs[:30]
        self.save = []
        self.grid_size = (
            (self._coordinate_max - self._coordinate_min) / self._map_resolution[0],
            (self._coordinate_max - self._coordinate_min) / self._map_resolution[1],
        )
        self.graph_share = getattr(self._config, 'GRAPH_SHARE', None)
        self.draw_curr_location = getattr(self._config, 'DRAW_CURR_LOCATION', 'point')
        self.record = True
        self.loose_check = False
        self.height_th = 0.1
        super().__init__()

    def to_grid(self, realworld_x, realworld_y):
        grid_x = int((self._coordinate_max - realworld_x) / self.grid_size[0])
        grid_y = int((realworld_y - self._coordinate_min) / self.grid_size[1])
        return grid_x, grid_y
    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "top_down_map"

    def _check_valid_nav_point(self, point: List[float]):
        self._sim.is_navigable(point)

    def get_original_map(self):
        top_down_map = get_topdown_map(
            self._sim,
            self._map_resolution,
            self._num_samples,
            self._config.DRAW_BORDER,
            loose_check=self.loose_check,
            height_th = self.height_th
        )

        range_x = np.where(np.any(top_down_map, axis=1))[0]
        range_y = np.where(np.any(top_down_map, axis=0))[0]

        if len(range_x) > 0 and len(range_y) > 0:
            self._ind_x_min = range_x[0]
            self._ind_x_max = range_x[-1]
            self._ind_y_min = range_y[0]
            self._ind_y_max = range_y[-1]
        else:
            top_down_map = get_topdown_map(
                self._sim,
                self._map_resolution,
                self._num_samples,
                self._config.DRAW_BORDER,
                loose_check=self.loose_check,
                height_th=self.height_th,
                draw_new_map=True
            )

            range_x = np.where(np.any(top_down_map, axis=1))[0]
            range_y = np.where(np.any(top_down_map, axis=0))[0]
            if len(range_x) > 0 and len(range_y) > 0:
                self._ind_x_min = range_x[0]
                self._ind_x_max = range_x[-1]
                self._ind_y_min = range_y[0]
                self._ind_y_max = range_y[-1]

        if self._config.FOG_OF_WAR.DRAW:
            self._fog_of_war_mask = np.zeros_like(top_down_map)

        return top_down_map

    def _draw_point(self, position, point_type, ch=None):
        t_x, t_y = to_grid(
            position[0],
            position[2],
            self._coordinate_min,
            self._coordinate_max,
            self._map_resolution,
        )
        if ch is None:
            self._top_down_map[
                t_x - self.point_padding : t_x + self.point_padding + 1,
                t_y - self.point_padding : t_y + self.point_padding + 1,
            ] = point_type
        else:
            self._top_down_map[
                t_x - self.point_padding : t_x + self.point_padding + 1,
                t_y - self.point_padding : t_y + self.point_padding + 1,
                ch
            ] = point_type

    def _draw_boundary(self, position, point_type):
        t_x, t_y = to_grid(
            position[0],
            position[2],
            self._coordinate_min,
            self._coordinate_max,
            self._map_resolution,
        )
        padd = int(self.point_padding/2)
        original = copy.deepcopy(        self._top_down_map[
            t_x - padd : t_x + padd  + 1,
            t_y - padd : t_y + padd  + 1,

        ])
        self._top_down_map[
            t_x - self.point_padding  - 1: t_x + self.point_padding + 2,
            t_y - self.point_padding  - 1 : t_y + self.point_padding + 2
        ] = point_type

        self._top_down_map[
        t_x - padd : t_x + padd  + 1,
        t_y - padd : t_y + padd  + 1
        ] = original

    def _draw_goals_view_points(self, episode):
        if self._config.DRAW_VIEW_POINTS:
            for goal in episode.goals:
                try:
                    if goal.view_points is not None:
                        for view_point in goal.view_points:
                            self._draw_point(
                                view_point.agent_state.position,
                                maps.MAP_VIEW_POINT_INDICATOR,
                            )
                except AttributeError:
                    pass

    def _draw_goals_positions(self, episode):
        if self._config.DRAW_GOAL_POSITIONS:

            for goal in episode.goals:
                try:
                    self._draw_point(
                        goal.position, maps.MAP_TARGET_POINT_INDICATOR
                    )
                except AttributeError:
                    pass

    def _draw_curr_goal_positions(self, goals, goal_mask=None):
        for goal in goals:
            self._draw_point(goal.position, maps.MAP_TARGET_POINT_INDICATOR)

    def _draw_goals_aabb(self, episode):
        if self._config.DRAW_GOAL_AABBS:
            for goal in episode.goals:
                try:
                    sem_scene = self._sim.semantic_annotations()
                    object_id = goal.object_id
                    assert int(
                        sem_scene.objects[object_id].id.split("_")[-1]
                    ) == int(
                        goal.object_id
                    ), f"Object_id doesn't correspond to id in semantic scene objects dictionary for episode: {episode}"

                    center = sem_scene.objects[object_id].aabb.center
                    x_len, _, z_len = (
                        sem_scene.objects[object_id].aabb.sizes / 2.0
                    )
                    # Nodes to draw rectangle
                    corners = [
                        center + np.array([x, 0, z])
                        for x, z in [
                            (-x_len, -z_len),
                            (-x_len, z_len),
                            (x_len, z_len),
                            (x_len, -z_len),
                            (-x_len, -z_len),
                        ]
                    ]

                    map_corners = [
                        to_grid(
                            p[0],
                            p[2],
                            self._coordinate_min,
                            self._coordinate_max,
                            self._map_resolution,
                        )
                        for p in corners
                    ]

                    maps.draw_path(
                        self._top_down_map,
                        map_corners,
                        maps.MAP_TARGET_BOUNDING_BOX,
                        self.line_thickness,
                    )
                except AttributeError:
                    pass

    def _draw_shortest_path(
        self, episode: Episode, agent_position: AgentState
    ):
        if self._config.DRAW_SHORTEST_PATH:
            self._shortest_path_points = self._sim.get_straight_shortest_path_points(
                agent_position, episode.goals[0].position
            )
            self._shortest_path_points = [
                to_grid(
                    p[0],
                    p[2],
                    self._coordinate_min,
                    self._coordinate_max,
                    self._map_resolution,
                )
                for p in self._shortest_path_points
            ]
            maps.draw_path(
                self._top_down_map,
                self._shortest_path_points,
                maps.MAP_SHORTEST_PATH_COLOR,
                self.line_thickness,
            )


    def _draw_path(self, p1, p2, color, ch=None):
        points = [to_grid(p1[0], p1[2], self._coordinate_min, self._coordinate_max, self._map_resolution),
                  to_grid(p2[0], p2[2], self._coordinate_min, self._coordinate_max, self._map_resolution)]
        maps.draw_path(
            self._top_down_map,
            points,
            color,
            self.line_thickness,
        )

    def reset_metric(self, *args: Any, episode, **kwargs: Any):
        self.node_list = None
        self._step_count = 0
        self._metric = None
        self.done_goals = []
        self.curr_goal = None
        if not self.record: return
        self._top_down_map = np.array(self.get_original_map())
        self._fog_of_war_mask = np.zeros_like(self._top_down_map)

        agent_state = self._sim.get_agent_state()
        agent_position = agent_state.position
        a_x, a_y = to_grid(
            agent_position[0],
            agent_position[2],
            self._coordinate_min,
            self._coordinate_max,
            self._map_resolution,
        )
        self._previous_xy_location = (a_y, a_x)
        self.update_fog_of_war_mask(np.array([a_x, a_y]), agent_state.rotation)
        self._draw_shortest_path(episode, agent_position)
        self._draw_goals_view_points(episode)
        self._draw_goals_aabb(episode)
        self._draw_goals_positions(episode)
        self._stored_map = copy.deepcopy(self._top_down_map)

        if self._config.DRAW_SOURCE:
            self._draw_point(
                episode.start_position, maps.MAP_SOURCE_POINT_INDICATOR
            )
        self.update_metric(episode, None)


        house_map, map_agent_x, map_agent_y = self.update_map(
            agent_state.position, agent_state.rotation
        )
        agent_map_coord = (map_agent_x - (self._ind_x_min - self._grid_delta),
                                 map_agent_y - (self._ind_y_min - self._grid_delta))
        polar_angle = self.get_polar_angle( agent_state.rotation)

        # Rather than return the whole map which may have large empty regions,
        # only return the occupied part (plus some padding).
        clipped_house_map = self._clip_map(house_map)

        clipped_fog_of_war_map = None
        if self._config.FOG_OF_WAR.DRAW:
            clipped_fog_of_war_map = self._clip_map(self._fog_of_war_mask)

        self._metric = {
            "map": clipped_house_map,
            "fog_of_war_mask": clipped_fog_of_war_map,
            "agent_map_coord": agent_map_coord,
            "agent_angle": polar_angle
        }

    def _clip_map(self, _map):
        min_x = max(self._ind_x_min - self._grid_delta, 0)
        min_y = max(self._ind_y_min - self._grid_delta, 0)
        return _map[
            min_x : self._ind_x_max + self._grid_delta,
            min_y: self._ind_y_max + self._grid_delta,
        ]

    def update_metric(self, episode, action, *args: Any, **kwargs: Any):
        if not self.record: return
        self._step_count += 1

        agent_state = self._sim.get_agent_state()
        house_map, map_agent_x, map_agent_y = self.update_map(
            agent_state.position, agent_state.rotation
        )
        agent_map_coord = (map_agent_x - (self._ind_x_min - self._grid_delta),
                                 map_agent_y - (self._ind_y_min - self._grid_delta))
        polar_angle = self.get_polar_angle( agent_state.rotation)

        clipped_house_map = self._clip_map(house_map)

        clipped_fog_of_war_map = None
        if self._config.FOG_OF_WAR.DRAW:
            clipped_fog_of_war_map = self._clip_map(self._fog_of_war_mask)#.max(axis=2)

        self._metric = {
            "map": clipped_house_map,
            "fog_of_war_mask": clipped_fog_of_war_map,
            "agent_map_coord": agent_map_coord,
            "agent_angle": polar_angle
        }
        self._top_down_map = copy.deepcopy(self._stored_map)


    def get_polar_angle(self, ref_rotation=None):
        if ref_rotation is None:
            agent_state = self._sim.get_agent_state()
            # quaternion is in x, y, z, w format
            ref_rotation = agent_state.rotation
        vq = np.quaternion(0,0,0,0)
        vq.imag = np.array([0,0,-1])
        heading_vector = (ref_rotation.inverse() * vq * ref_rotation).imag
        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        x_y_flip = -np.pi / 2
        return np.array(phi) + x_y_flip

    def update_map(self, agent_position, agent_rotation=None):
        a_x, a_y = to_grid(
            agent_position[0],
            agent_position[2],
            self._coordinate_min,
            self._coordinate_max,
            self._map_resolution,
        )
        self.update_fog_of_war_mask(np.array([a_x, a_y]), agent_rotation)

        self._previous_xy_location = (a_y, a_x)
        return self._top_down_map, a_x, a_y

    def update_graph(self, node_list, affinity, graph_mask, curr_info={}, flags=None, goal_id=None):
        self.node_list = node_list
        draw_point_list = []

        for idx, node_position in enumerate(self.node_list):
            neighbors = torch.where(affinity[idx])[0]
            node_color_index = COMMON_NODE
            edge_color_index = COMMON_EDGE
            for neighbor_idx in neighbors:
                neighbor_position = node_list[neighbor_idx]
                self._draw_path(node_position, neighbor_position, edge_color_index)
            draw_point_list.append([node_position, node_color_index])

        for node_position, node_color_index in draw_point_list:
            self._draw_point(node_position, node_color_index)
        self._draw_boundary(self.node_list[curr_info['curr_node']], CURR_NODE)
        self.curr_info = curr_info

    def update_fog_of_war_mask(self,agent_position, agent_rotation=None):
        if self._config.FOG_OF_WAR.DRAW:
            self._fog_of_war_mask = fog_of_war.reveal_fog_of_war(
                self._top_down_map,
                self._fog_of_war_mask,
                agent_position,
                self.get_polar_angle(agent_rotation),
                fov=self._config.FOG_OF_WAR.FOV,
                max_line_len=self._config.FOG_OF_WAR.VISIBILITY_DIST
                * max(self._map_resolution)
                / (self._coordinate_max - self._coordinate_min),
            )
