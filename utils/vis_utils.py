
import os
import textwrap
from typing import Dict, List, Optional, Tuple

import imageio
import numpy as np
import tqdm

from habitat.core.logging import logger
from habitat.core.utils import try_cv2_import
from habitat.utils.visualizations import maps
import cv2

import scipy
from habitat.utils.visualizations import utils

agent_colors = ['red','blue', 'yellow', 'green']
AGENT_IMGS = []
for color in agent_colors:
    img = np.ascontiguousarray(np.flipud(imageio.imread('env_utils/agent_pictures/agent_{}.png'.format(color))))
    AGENT_IMGS.append(img)

def draw_agent(
    image: np.ndarray,
    agent_id,
    agent_center_coord: Tuple[int, int],
    agent_rotation: float,
    agent_radius_px: int = 5,
) -> np.ndarray:
    r"""Return an image with the agent image composited onto it.
    Args:
        image: the image onto which to put the agent.
        agent_center_coord: the image coordinates where to paste the agent.
        agent_rotation: the agent's current rotation in radians.
        agent_radius_px: 1/2 number of pixels the agent will be resized to.
    Returns:
        The modified background image. This operation is in place.
    """

    # Rotate before resize to keep good resolution.
    rotated_agent = scipy.ndimage.interpolation.rotate(
        AGENT_IMGS[agent_id], agent_rotation * 180 / np.pi
    )
    # Rescale because rotation may result in larger image than original, but
    # the agent sprite size should stay the same.
    initial_agent_size = AGENT_IMGS[agent_id].shape[0]
    new_size = rotated_agent.shape[0]
    agent_size_px = max(
        1, int(agent_radius_px * 2 * new_size / initial_agent_size)
    )
    resized_agent = cv2.resize(
        rotated_agent,
        (agent_size_px, agent_size_px),
        interpolation=cv2.INTER_LINEAR,
    )
    utils.paste_overlapping_image(image, resized_agent, agent_center_coord)
    return image


def clip_map_birdseye_view(image, clip_size, pixel_pose):
    half_clip_size = clip_size//2

    delta_x = pixel_pose[0] - half_clip_size
    delta_y = pixel_pose[1] - half_clip_size
    min_x = max(delta_x, 0)
    max_x = min(pixel_pose[0] + half_clip_size, image.shape[0])
    min_y = max(delta_y, 0)
    max_y = min(pixel_pose[1] + half_clip_size, image.shape[1])

    return_image = np.zeros([clip_size, clip_size, 3],dtype=np.uint8)
    cliped_image = image[min_x:max_x, min_y:max_y]
    start_x = max(-delta_x,0)
    start_y = max(-delta_y,0)
    try:
        return_image[start_x:start_x+cliped_image.shape[0],start_y:start_y+cliped_image.shape[1]] = cliped_image
    except:
        print('image shape ', image.shape, 'min_x', min_x,'max_x', max_x,'min_y',min_y,'max_y',max_y, 'return_image.shape',return_image.shape, 'cliped', cliped_image.shape, 'start_x,y', start_x, start_y)
    return return_image


def append_text_to_image(image: np.ndarray, text: str, font_size=0.5, font_line=cv2.LINE_AA):
    r""" Appends text underneath an image of size (height, width, channels).
    The returned image has white text on a black background. Uses textwrap to
    split long text into multiple lines.
    Args:
        image: the image to put text underneath
        text: a string to display
    Returns:
        A new image with text inserted underneath the input image
    """
    h, w, c = image.shape
    font_thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    blank_image = np.zeros(image.shape, dtype=np.uint8)
    linetype = font_line if font_line is not None else cv2.LINE_8

    char_size = cv2.getTextSize(" ", font, font_size, font_thickness)[0]
    wrapped_text = textwrap.wrap(text, width=int(w / char_size[0]))

    y = 0
    for line in wrapped_text:
        textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
        y += textsize[1] + 10
        if y % 2 == 1 :
            y += 1
        x = 10
        cv2.putText(
            blank_image,
            line,
            (x, y),
            font,
            font_size,
            (255, 255, 255),
            font_thickness,
            lineType=linetype,
        )
    text_image = blank_image[0 : y + 10, 0:w]
    final = np.concatenate((image, text_image), axis=0)
    return final
def draw_collision(view: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    r"""Draw translucent red strips on the border of input view to indicate
    a collision has taken place.
    Args:
        view: input view of size HxWx3 in RGB order.
        alpha: Opacity of red collision strip. 1 is completely non-transparent.
    Returns:
        A view with collision effect drawn.
    """
    strip_width = view.shape[0] // 20
    mask = np.ones(view.shape)
    mask[strip_width:-strip_width, strip_width:-strip_width] = 0
    mask = mask == 1
    view[mask] = (alpha * np.array([255, 0, 0]) + (1.0 - alpha) * view)[mask]
    return view


def observations_to_image(observation: Dict, info: Dict, mode='panoramic', local_imgs=None, clip=None, center_agent = True) -> np.ndarray:
    r"""Generate image of single frame from observation and info
    returned from a single environment step().

    Args:
        observation: observation returned from an environment step().
        info: info returned from an environment step().

    Returns:
        generated image of a single frame.
    """
    size = 2.0
    egocentric_view = []
    if "rgb" in observation and mode != 'panoramic':
        rgb = observation["rgb"]
        if not isinstance(rgb, np.ndarray):
            rgb = rgb.cpu().numpy()
        rgb = cv2.putText(np.ascontiguousarray(rgb), 'current_obs',(5,10),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255))
        egocentric_view.append(rgb)
    elif "panoramic_rgb" in observation and mode == 'panoramic':
        rgb = observation['panoramic_rgb']
        if not isinstance(rgb, np.ndarray):
            rgb = rgb.cpu().numpy()
        rgb = cv2.putText(np.ascontiguousarray(rgb), 'current_obs',(5,10),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255))
        egocentric_view.append(rgb)

    if "target_goal" in observation and len(observation['target_goal']) > 0:
        goal_rgb = (observation['target_goal']*255)
        if not isinstance(goal_rgb, np.ndarray):
            goal_rgb = goal_rgb.cpu().numpy()
        if len(goal_rgb.shape) == 4:
            if info is not None:
                goal_rgb = goal_rgb * (1 - info['total_success']).reshape(-1, *[1] * len(goal_rgb.shape[1:]))
            goal_rgb = np.concatenate(np.split(goal_rgb[:,:,:,:3],goal_rgb.shape[0],axis=0),1).squeeze(axis=0)
        else:
            goal_rgb = goal_rgb[:,:,:3]
        goal_rgb = cv2.putText(np.ascontiguousarray(goal_rgb), 'target_obs',(5,10),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255))
        egocentric_view.append(goal_rgb.astype(np.uint8))

    if len(egocentric_view) > 0:
        if mode == 'panoramic':
            egocentric_view = np.concatenate(egocentric_view, axis=0)
        else:
            egocentric_view = np.concatenate(egocentric_view, axis=1)
        if "collisions" in info and info['collisions'] is not None:
            if info["collisions"]["is_collision"]:
                egocentric_view = draw_collision(egocentric_view)
        frame = cv2.resize(egocentric_view, dsize=None, fx=size*0.75, fy=size)
    else:
        frame = None

    if info is not None and "top_down_map" in info:
        if info['top_down_map'] is not None:
            top_down_height = frame.shape[0] if frame is not None else info["top_down_map"]["map"].shape[0]
            top_down_map = info["top_down_map"]["map"]
            map_agent_pos = info["top_down_map"]["agent_map_coord"]

            color_top_down_map = maps.colorize_topdown_map(
                top_down_map, info["top_down_map"]["fog_of_war_mask"]
            )
            top_down_map = draw_agent(
                image=color_top_down_map,
                agent_id=1,
                agent_center_coord=map_agent_pos,
                agent_rotation=info["top_down_map"]["agent_angle"],
                agent_radius_px=5,
            )

            # scale top down map to align with rgb view
            old_h, old_w, _ = top_down_map.shape
            top_down_width = int(float(top_down_height) / old_h * old_w)
            # cv2 resize (dsize is width first)
            if frame is not None:
                top_down_map = cv2.resize(
                    top_down_map,
                    (top_down_width, top_down_height),
                    interpolation=cv2.INTER_CUBIC,
                )
        else:
            height = frame.shape[0] if frame is not None else 512
            top_down_map = np.zeros([height, height, 3],dtype=np.uint8)

        frame = np.concatenate((frame, top_down_map), axis=1) if frame is not None else top_down_map

    return frame
