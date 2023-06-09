import numpy as np

color_dict = {
    'red': [255, 0, 0],
    'orange': [255, 100, 0],
    'yellow': [255, 200, 0],
    'green': [0, 150, 0],
    'blue': [0, 100, 255],
    'purple': [100, 0, 255],
    'black': [0, 0, 0],
    'white': [255, 255, 255],
}
colors_rgb = np.array(list(color_dict.values())).astype(np.uint8)
colors = list(color_dict.keys())
