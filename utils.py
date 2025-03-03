from copy import deepcopy
from hashlib import md5
from os import system
from shutil import which
from subprocess import check_output

import cv2
import numpy as np
from attrs import asdict, define
from numpy import uint8
from numpy.typing import NDArray
from PIL.Image import Image, alpha_composite, fromarray
from streamlit import code, expander, get_option, toast
from streamlit.delta_generator import DeltaGenerator
from streamlit.elements.lib.image_utils import image_to_url
from streamlit_drawable_canvas import CanvasResult, _component_func, _data_url_to_image  # noqa: PLC2701
from supervision import Color, ColorLookup, ColorPalette, Point, Position
from ultralytics import RTDETR, SAM, YOLO, FastSAM, YOLOWorld
from vidgear.gears import VideoGear

i8 = NDArray[uint8]

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

base_colors = ['black', 'white']


class ColorClassifier:
  __slots__ = ('names', 'rgb', 'ycc')

  def __init__(self, names: list[str] | None = None) -> None:
    if names is None:
      names = base_colors
    d = {k: v for k, v in color_dict.items() if k in names}
    self.names = list(d.keys())

    if self.names:
      rgb_mat = np.array(list(d.values())).astype(uint8)
      self.ycc = rgb2ycc(rgb_mat)
      self.rgb = [tuple(map(int, i)) for i in rgb_mat]

  def closest(self, _rgb: i8) -> int:
    return np.argmin(np.sum((self.ycc - rgb2ycc(_rgb[np.newaxis])) ** 2, axis=1))


@define
class Draw:
  lines: list = []  # noqa: RUF008
  zones: list = []  # noqa: RUF008

  def __str__(self) -> str:
    s = ''
    if l := len(self.lines):  # noqa: E741
      s += '\n - ' + plur(l, 'line')
    if z := len(self.zones):
      s += '\n - ' + plur(z, 'zone')
    return s

  def __len__(self) -> int:
    return len(self.lines) + len(self.zones)

  @classmethod
  def from_canvas(cls, d: list) -> 'Draw':
    return cls(
      lines=[
        (
          (i['left'] + i['x1'], i['top'] + i['y1']),
          (i['left'] + i['x2'], i['top'] + i['y2']),
        )
        for i in d
        if i['type'] == 'line'
      ],
      zones=[[[x[1], x[2]] for x in k] for k in [j[:-1] for j in [i['path'] for i in d if i['type'] == 'path']]]
      + [
        [
          [i['left'], i['top']],
          [i['left'] + i['width'], i['top']],
          [i['left'] + i['width'], i['top'] + i['height']],
          [i['left'], i['top'] + i['height']],
        ]
        for i in d
        if i['type'] == 'rect'
      ],
    )


def weight2model(w: str) -> YOLO | YOLOWorld | SAM | RTDETR | FastSAM:
  if w.startswith('yolo'):
    return YOLOWorld(w) if 'world' in w else YOLO(w)
  if w.startswith(('sam', 'mobile_sam')):
    return SAM(w)
  if w.startswith('rtdetr'):
    return RTDETR(w)
  if w.startswith('FastSAM'):
    return FastSAM(w)
  msg = 'Unknown model'
  raise ValueError(msg)


def available_reso() -> list[tuple[int, int]]:
  cmd = ''
  if which('v4l2-ctl'):
    cmd = "v4l2-ctl -d /dev/video0 --list-formats-ext | grep x | awk '{print $NF}' | sort -u"

  if which('system_profiler'):
    dev = 'FaceTime HD Camera'
    if dev in check_output('system_profiler SPCameraDataType', shell=True).decode().strip():
      cmd = f"ffmpeg -hide_banner -f avfoundation -r 1 -i '{dev}' 2>&1 >/dev/null | grep -oE '{r'[0-9]{3,4}x[0-9]{3,4}'}' | sort -u"  # noqa: E501

  return sorted(
    [
      i
      for i in [tuple(map(int, i.split('x'))) for i in check_output(cmd, shell=True).decode().strip().splitlines()]
      if i[0] > i[1] and len(i) == 2  # noqa: PLR2004
    ],
    key=lambda x: x[0] * x[1],
  )


def plur(n: int, s: str) -> str:
  return f'{n} {s}{"s"[: n ^ 1]}' if n else ''


def rgb2hex(rgb: tuple[int, int, int]) -> str:
  r, g, b = rgb
  return f'#{r:02x}{g:02x}{b:02x}'


def rgb2ycc(rgb: i8) -> i8:
  rgb_norm = rgb / 255.0
  r, g, b = rgb_norm[:, 0], rgb_norm[:, 1], rgb_norm[:, 2]
  y = 0.299 * r + 0.587 * g + 0.114 * b
  cb = 128 - 0.168736 * r - 0.331364 * g + 0.5 * b
  cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
  return np.stack([y, cb, cr], axis=-1)


def avg_rgb(f: i8) -> i8:
  return cv2.kmeans(
    f.reshape(-1, 3).astype(np.float32)[..., ::-1],
    1,
    None,
    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
    10,
    cv2.KMEANS_RANDOM_CENTERS,
  )[2][0].astype(uint8)


def filter_by_vals(place: DeltaGenerator, text: str, d: dict[int, str]) -> list[int | str]:
  a = list(d.values())
  return (
    [a.index(i) for i in place.multiselect(' ', a, label_visibility='collapsed')]
    if place.toggle(text)
    else list(d.keys())
  )


def exe_button(place: DeltaGenerator, cmd: str, text: str, noti: str, tips: str | None = None) -> None:
  if place.button(text, help=tips):
    code(cmd, language='bash')
    toast(noti)
    system(cmd)


# remove this when the library get maintained


def _resize_img(img: Image, new_height: int = 700, new_width: int = 700) -> Image:
  h_ratio = new_height / img.height
  w_ratio = new_width / img.width
  return img.resize((int(img.width * w_ratio), int(img.height * h_ratio)))


def st_canvas(
  fill_color: str = '#eee',
  stroke_width: int = 20,
  stroke_color: str = 'black',
  background_color: str = '',
  background_image: Image | None = None,
  update_streamlit: bool = True,  # noqa: FBT001, FBT002
  height: int = 400,
  width: int = 600,
  drawing_mode: str = 'freedraw',
  point_display_radius: int = 3,
  key: str | None = None,
) -> CanvasResult:
  background_image_url = None
  if background_image:
    background_image = _resize_img(background_image, height, width)
    background_image_url = image_to_url(
      background_image,
      width,
      True,  # noqa: FBT003
      'RGB',
      'PNG',
      f'drawable-canvas-bg-{md5(background_image.tobytes()).hexdigest()}-{key}',  # noqa: S324
    )
    background_image_url = get_option('server.baseUrlPath') + background_image_url
    background_color = ''

  component_value = _component_func(
    fillColor=fill_color,
    strokeWidth=stroke_width,
    strokeColor=stroke_color,
    backgroundColor=background_color,
    backgroundImageURL=background_image_url,
    realtimeUpdateStreamlit=update_streamlit and (drawing_mode != 'polygon'),
    canvasHeight=height,
    canvasWidth=width,
    drawingMode=drawing_mode,
    initialDrawing=None,
    displayToolbar=True,
    displayRadius=point_display_radius,
    key=key,
    default=None,
  )
  if component_value is None:
    return CanvasResult()

  return CanvasResult(
    np.asarray(_data_url_to_image(component_value['data'])),
    component_value['raw'],
  )


def canvas(stroke: str, width: int, height: int, mode: str, bg: Image | None, key: str) -> CanvasResult:
  return st_canvas(
    stroke_width=2,
    fill_color='#ffffff55',
    stroke_color=stroke,
    width=width,
    height=height,
    drawing_mode=mode,
    background_image=bg,
    key=key,
  )


def canvas2draw(wh: tuple[int, int], background: Image, *, key: str, line: bool) -> Draw:
  width, height = wh
  ex = expander('Draw on canvas')
  modes = (['line'] if line else []) + ['rect', 'polygon']
  mode = ex.selectbox(' ', modes, label_visibility='collapsed')
  mt = ex.empty()
  enable_bg, bg = False, None
  if background:
    enable_bg = ex.toggle('Background', value=True)
    bg = background if enable_bg else None
  stroke = '#fff' if bg is None else '#000'
  cv = canvas(stroke, width, height, mode, bg, key + str(enable_bg))

  draw = Draw()
  if cv.json_data is not None:
    draw = Draw.from_canvas(cv.json_data['objects'])
    mt.markdown(draw)
  if cv.image_data is not None and len(draw) > 0 and bg and ex.button('Export canvas'):
    alpha_composite(bg.convert('RGBA'), fromarray(cv.image_data)).save('canvas.png')
  return draw


def first_frame(path: str) -> Image:
  stream = VideoGear(source=path).start()
  frame = fromarray(stream.read()[..., ::-1])
  stream.stop()
  return frame


def to_plain(ori: dict) -> dict:
  d = deepcopy(ori)
  for v in d.values():
    for k2, v2 in v.items():
      match v2:
        case ColorLookup():
          v[k2] = v2.value
        case Position():
          v[k2] = v2.value
        case Color():
          v[k2] = rgb2hex(v2.as_rgb())
        case Point():
          v[k2] = v2.as_xy_int_tuple()
        case ColorClassifier():
          v[k2] = v2.names
        case ColorPalette():
          v[k2] = None
        case Draw():
          v[k2] = asdict(v2)

  for k, v in d.items():
    d[k] = {k2: v2 for k2, v2 in v.items() if v2 is not None}

  return d


def from_plain(ori: dict) -> dict:
  d = deepcopy(ori)
  for v in d.values():
    for k2, v2 in v.items():
      match k2:
        case str(k2) if 'lookup' in k2:
          v[k2] = ColorLookup(v2)
        case str(k2) if 'position' in k2 and k2 != 'smart_position':
          v[k2] = Position(v2)
        case str(k2) if 'anchor' in k2:
          v[k2] = Point(v2[0], v2[1])
        case str(k2) if 'color' in k2:
          v[k2] = Color.from_hex(v2)
        case str(k2) if 'names' in k2:
          v[k2] = {int(k3): v3 for k3, v3 in v2.items()}
        case str(k2) if 'draw' in k2:
          v[k2] = Draw(**v2)

  return d


def unsnake(s: str) -> str:
  return s.replace('_', ' ').capitalize()
