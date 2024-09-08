from copy import deepcopy
from glob import glob
from inspect import signature
from json import dump
from os.path import isfile
from pathlib import Path
from shutil import which
from time import gmtime, strftime, time
from typing import Any

from av.video.frame import VideoFrame
from cv2 import VideoCapture, VideoWriter_fourcc, destroyAllWindows
from PIL import Image
from streamlit import camera_input, empty, markdown, session_state, set_page_config, toast
from streamlit import sidebar as sb
from streamlit_webrtc import webrtc_streamer
from streamlit_webrtc.models import VideoFrameCallback
from supervision import ColorLookup, Position, VideoInfo
from torch.cuda import empty_cache
from ultralytics.utils.downloads import GITHUB_ASSETS_NAMES

from core import Annotator, Model
from custom_annotator import all_class
from utils import (
  ColorClassifier,
  available_reso,
  canvas2draw,
  color_dict,
  exe_button,
  filter_by_vals,
  first_frame,
  i8,
  rgb2hex,
  to_plain,
  unsnake,
  weight2model,
)

set_page_config(
  page_icon='🎥',
  page_title='StreamlitVision',
  layout='wide',
  initial_sidebar_state='expanded',
  menu_items={'Report a bug': 'https://github.com/1qh/StreamlitVision/issues/new'},
)
with open('style.css', encoding='utf-8') as css:
  markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

total_frames = 1

all_names = list(all_class.keys())
all_default = {}

for i in all_names:
  sig: dict[str, Any] = {}
  for j in signature(all_class[i]).parameters.items():
    sig |= {j[0]: j[1].default}
  all_default[i] = sig

custom_defaults = dict(text_padding=1, text_thickness=1, thickness=1)

for each_default in all_default.values():
  for k_conf, v_conf in custom_defaults.items():
    if k_conf in each_default:
      each_default[k_conf] = v_conf


all_plain = to_plain(all_default)

_shape: tuple[int, ...] | None = None

if 'source' not in session_state:
  session_state['source'] = ''

run_toggle = sb.empty()
enable_camera = sb.toggle('Use camera')
file = sb.file_uploader(' ', label_visibility='collapsed')

running = False
if file or enable_camera:
  running = run_toggle.toggle('Realtime inference', help='Slower than native')


m_tab, an_tab, vid_tab, exp_tab = sb.tabs(['Model', 'Annotate', 'Camera' if enable_camera else 'Video', 'Export'])
fps_slot = sb.empty()

pretrained = [i for i in GITHUB_ASSETS_NAMES if i.endswith('.pt') and '_nas_' not in i]
custom = [i for i in glob('*.pt') if i not in pretrained]

use_custom = m_tab.toggle(
  'Custom weight',
  disabled=not custom,
  help=f"{'' if len(custom) else 'No custom weight found '}in current folder",
)

prefixes = {
  'YOLO': 'yolo',
  'RTDETR': 'rtdetr',
  'FastSAM': 'FastSAM',
  'SAM': ('sam', 'mobile_sam'),
}
if not use_custom and m_tab.toggle('Filter weight', value=True):
  k = m_tab.radio(' ', ['YOLO', 'RTDETR', 'FastSAM', 'SAM'], label_visibility='collapsed', horizontal=True)
  pretrained = [i for i in pretrained if i.startswith(prefixes[k])]

weight = m_tab.selectbox(' ', custom if use_custom else pretrained, label_visibility='collapsed')
_model = weight2model(weight)

tracker = m_tab.selectbox('Tracker', ['bytetrack', 'botsort', 'No track']) if _model.task != 'classify' else None
tracker = tracker if tracker != 'No track' else None

conf = m_tab.slider('Threshold', max_value=1.0, value=0.25)
iou = m_tab.slider('IoU', max_value=1.0, value=0.45)

classes = (
  filter_by_vals(m_tab, 'Custom Classes', _model.names)
  if hasattr(_model, 'names') and isinstance(_model.names, dict) and len(_model.names.values()) > 1
  else []
)

options: dict[str, Any] = dict(conf=conf, iou=iou)
if len(classes):
  options |= dict(classes=classes)
if tracker:
  options |= dict(tracker=f'{tracker}.yaml', persist=True)

model = Model(weight, options)
an = Annotator(model)

infer = empty()
progress_bar = empty()
source = ''
wh = (0, 0)
canvas_bg = None
draw = None
task = model.task

if enable_camera:
  session_state['source'] = 0
  file = None
  resos = available_reso()
  wh = vid_tab.selectbox('Resolution', resos) if len(resos) else (1280, 720)
  if vid_tab.toggle('Annotate from image'):
    if vid_tab.toggle('Upload'):
      canvas_bg = vid_tab.file_uploader(' ', label_visibility='collapsed', key='u')
    if vid_tab.toggle('Shoot'):
      if shot := camera_input('Shoot'):
        model.predict_image(shot)
        canvas_bg = Image.open(shot)
  draw = canvas2draw(wh, canvas_bg, str(wh[0]), bool(tracker)) if task in ('detect', 'segment') else None

if file:
  if 'image' in file.type:
    model.predict_image(file)

  elif 'video' in file.type:
    vid_tab.video(file)
    source = f'up_{file.name}'

    with open(source, 'wb') as up:
      up.write(file.read())

    session_state['source'] = source

    vid = VideoInfo.from_video_path(source)

    if vid.total_frames and which('ffmpeg') and vid_tab.toggle('Trim'):
      length = int(vid.total_frames / vid.fps)
      sec_begin, sec_end = vid_tab.slider(' ', value=(0, length), max_value=length, label_visibility='collapsed')
      begin, end = [strftime('%M:%S', gmtime(t)) for t in [sec_begin, sec_end]]
      if sec_end - sec_begin < length:
        trim = f'trim_{source[3:]}'
        if isfile(trim):
          session_state['source'] = trim
        exe_button(
          vid_tab,
          f'ffmpeg -y -i {source} -ss {begin} -to {end} -c copy {trim}',
          f'Trim from {begin} to {end}',
          'Trimming...',
        )

    source = session_state['source']
    final_vid = VideoInfo.from_video_path(source)
    wh = final_vid.resolution_wh
    total_frames = final_vid.total_frames or total_frames

    vid_tab.markdown(f"""
- Video resolution: {'x'.join([str(i) for i in wh])}
- Total frames: {total_frames}
- FPS: {final_vid.fps}
- Path: {source}
""")
    draw = canvas2draw(wh, first_frame(source), source, tracker) if task in ('detect', 'segment') else None

  else:
    sb.warning('Please upload image/video')


base_anns = {'Fps', 'Label'}

match task:
  case 'detect':
    base_anns.add('BoxCorner')
  case 'segment':
    base_anns.add('Halo')
  case 'pose':
    base_anns.add('Edge')
    base_anns.remove('Label')
  case _:
    pass

if classes:
  base_anns.add('Count')

if tracker:
  base_anns.add('Trace')

if draw:
  base_anns.add('LineAndZone')

ann_names = an_tab.multiselect(' ', all_names, base_anns, label_visibility='collapsed')

# config_plain = all_plain
origin_config_plain: dict[str, dict[str, Any]] = {k: v for k, v in all_plain.items() if k in ann_names}

config = deepcopy(origin_config_plain)

for k, v in config.items():
  ex = an_tab.expander(k)

  ini_conf = {}
  for k2, v2 in v.items():
    key = f'{k}_{k2}'
    ini_conf[key] = list(v2) if isinstance(v2, tuple) else v2

    tit = unsnake(k2)
    match k2:
      case str(k2) if 'lookup' in k2:
        lookup_list = ColorLookup.list()
        v[k2] = ex.selectbox(tit, lookup_list, lookup_list.index(v2), key=key)

      case str(k2) if 'anchor' in k2:
        ex.subheader('Position')
        v[k2] = list(v[k2])
        v[k2][0] = ex.slider('x', 0, wh[0], v2[0], 1, key=f'{key}_x')
        v[k2][1] = ex.slider('y', 0, wh[1], v2[1], 1, key=f'{key}_y')

      case str(k2) if 'position' in k2:
        pos_list = Position.list()
        v[k2] = ex.selectbox(tit, pos_list, pos_list.index(v2), key=key)

      case str(k2) if '_color' in k2:
        v[k2] = ex.color_picker(tit, v2, key=key)

    match v2:
      case bool():
        v[k2] = ex.toggle(tit, v2, key=key)
      case int():
        abso = abs(v2)
        min_val = min([0, v2, 10 * v2 + 1])
        max_val = max([0, abso, 10 * abso + 1])
        v[k2] = ex.number_input(tit, min_val, max_val, v2, 1, key=key)
      case float():
        v[k2] = ex.number_input(tit, 0.0, 10 * v2 + 1.0, v2, 0.1, key=key)
      case dict():
        pass

  cur_conf = {f'{k}_{ki}': va for ki, va in v.items()}

  if cur_conf != ini_conf:
    diff = {k: v for k, v in ini_conf.items() if v != cur_conf[k]}
    for kd, vd in diff.items():
      ex.markdown(f'Default {unsnake(kd.removeprefix(k))} = {vd}')

  match k:
    case 'ColorClassifier':
      clf = ColorClassifier()
      ex.subheader('Classes')
      all_colors = color_dict.keys()
      color_names = (
        ex.multiselect(' ', all_colors, ['black', 'white']) if ex.toggle('Custom colors', value=True) else all_colors
      )
      if len(color_names) > 0:
        clf = ColorClassifier(color_names)
        for c, rgb in zip(clf.names, clf.rgb):
          ex.color_picker(f'{c}', value=rgb2hex(rgb))
      v['clf'] = clf
    case 'Count':
      v['names'] = model.names
    case 'LineAndZone':
      if draw:
        v['draw']['lines'] = draw.lines
        v['draw']['zones'] = draw.zones
      v['wh'] = wh

an = Annotator(model) if task == 'classify' else Annotator(model, config, wh)

source = session_state['source']
cap = VideoCapture(source)
if source == 0:
  width, height = wh
  codec = VideoWriter_fourcc(*'MJPG')
  cap.set(6, codec)
  cap.set(5, 30)
  if sum(wh) > 1:
    cap.set(3, width)
    cap.set(4, height)

count = 0

while running:
  f: i8
  success, f = cap.read()
  if not success:
    break
  prev_time = time()

  from_ann, fallback = an(f)
  t1, t2 = infer.tabs(['Main', 'Fallback'])
  t1.image(from_ann, use_column_width='always', channels='BGR')
  t2.image(fallback, use_column_width='always', channels='BGR')
  if source:
    count += 1
    progress_bar.progress(count / total_frames)

  curr_time = time()
  fps = 1 / (curr_time - prev_time)
  prev_time = curr_time
  fps_slot.metric('FPS', f'{fps:.2f}')

cap.release()


def stream(stream_key: str, callback: VideoFrameCallback):
  webrtc_streamer(
    key=stream_key,
    video_frame_callback=callback,
    media_stream_constraints={
      'video': {
        'width': {'min': width},
        'height': {'min': height},
      }
    },
  )


def cam(frame: VideoFrame):
  fr = frame.to_ndarray(format='rgb24')
  return VideoFrame.from_ndarray(model(fr)[1])


# oh my god, it took me so long to realize the frame bigger through time
def cam_adapt(frame: VideoFrame):
  fr = frame.to_ndarray(format='rgb24')
  global _shape
  if an.linezone and fr.shape != _shape:
    _shape = fr.shape
    an.linezone.update(fr)
  return VideoFrame.from_ndarray(an(fr)[0])


if enable_camera:
  if task == 'classify':
    stream('clf', cam)
  else:
    stream('dsp', cam_adapt)


if source != '':
  cmd = f'{Path(__file__).parent}/native.py --source {source}'
  tips = 'Export to update latest drawings & annotators before running in OpenCV'
  if exp_tab.button('Export config to JSON'):
    with open('config.json', 'w', encoding='utf-8') as json_file:
      dump(dict(config=config, options=options, weight=weight, wh=wh), json_file, indent=2)
    toast('Exported to `config.json`')

  if source != 0:
    output = exp_tab.text_input(' ', 'result.mp4', label_visibility='collapsed')
    exe_button(exp_tab, f'{cmd} --output {output}', f'Save to {output}', f'Saving to `{output}`...', tips=tips)
    exp_tab.markdown("<p style='text-align: center'>or</p>", unsafe_allow_html=True)
  exe_button(exp_tab, cmd, 'Inference in OpenCV', 'Press `q` to exit', tips=tips)


else:
  msg = 'Upload or Turn on camera'
  vid_tab.warning(msg)
  exp_tab.warning(msg)

empty_cache()
destroyAllWindows()
