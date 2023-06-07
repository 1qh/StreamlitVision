import os
from glob import glob
from pathlib import Path
from shutil import which
from time import gmtime, strftime

import cv2
import numpy as np
import streamlit as st
import yolov5
from av import VideoFrame
from lightning import LightningApp, LightningFlow
from lightning.app.frontend import StreamlitFrontend
from PIL import Image
from psutil import process_iter
from streamlit import session_state, set_page_config
from streamlit import sidebar as sb
from streamlit_webrtc import webrtc_streamer
from supervision import BoxAnnotator, Detections, VideoInfo
from ultralytics import RTDETR, YOLO
from vidgear.gears import VideoGear

from core import Annotator, Model, ModelInfo, cvt, maxcam

_shape = None

if 'path' not in session_state:
    session_state['path'] = ''


def st_config():
    set_page_config(
        page_icon='🎥',
        page_title='ComputerVisionWebUI',
        layout='wide',
        initial_sidebar_state='expanded',
        menu_items={
            'Report a bug': 'https://github.com/1qh/ComputerVisionWebUI/issues/new',
        },
    )
    st.markdown(
        """
    <style>
    div.block-container {padding-top:2rem}
    footer {visibility: hidden;}
    @font-face {font-family: 'SF Pro Display';}
    html, body, [class*="css"]  {font-family: 'SF Pro Display';}
    thead tr th:first-child {display:none}
    tbody th {display:none}
    </style>
    """,
        unsafe_allow_html=True,
    )


def custom_classes(model):
    d = model.model.names
    all = list(d.values())

    if sb.checkbox('Custom Classes'):
        return [
            all.index(i) for i in sb.multiselect(' ', all, label_visibility='collapsed')
        ]
    else:
        return list(d.keys())


def load():
    family = sb.radio('Model family', ('YOLO', 'RT-DETR'), horizontal=True)
    if family == 'YOLO':
        suffix = {
            'Detect': '',
            'Segment': '-seg',
            'Classify': '-cls',
            'Pose': '-pose',
        }
        tasks = list(suffix.keys())

        c0, c1, c2 = sb.columns(3)
        ver = c0.selectbox('Version', ('v8', 'v6', 'v5u', 'v5', 'v3'))
        if ver in ('v3', 'v5u', 'v6'):
            tasks = tasks[:1]
        elif ver == 'v5':
            tasks = tasks[:3]

        custom = sb.checkbox('Custom weight')
        task = c1.selectbox(
            'Task',
            tasks,
            disabled=custom and ver == 'v8',
        )
        size = c2.selectbox(
            'Size',
            ('n', 's', 'm', 'l', 'x'),
            disabled=custom or ver == 'v3',
        )
        if custom:
            path = sb.selectbox(' ', glob('*.pt'), label_visibility='collapsed')
        else:
            path = f"yolo{ver[:2]}{size if ver != 'v3' else ''}{suffix[task]}{ver[2] if len(ver) > 2 else ''}.pt"

        if ver == 'v5':
            model = yolov5.load(path)
            task = task.lower()
        else:
            model = YOLO(path)
            task = model.overrides['task']
            if custom:
                sb.success(f'{task.capitalize()} weights loaded')
            path = model.ckpt_path

    elif family == 'RT-DETR':
        ver = 'rtdetr'
        task = 'detect'
        size = sb.selectbox('Size', ('l', 'x'))
        path = f'{ver}-{size}.pt'
        model = RTDETR(path)

    conf = sb.slider('Threshold', max_value=1.0, value=0.25)
    classes = custom_classes(model)

    tracker = None
    if task != 'classify' and ver not in ('v5', 'rtdetr'):
        tracker = sb.selectbox('Tracker', ['bytetrack', 'botsort', None])

    return ModelInfo(
        path=path,
        classes=classes,
        ver=ver,
        task=task,
        conf=conf,
        tracker=tracker,
    )


def hms(s):
    return strftime('%H:%M:%S', gmtime(s))


def plot(i):
    return cvt(i.plot())


def trim_vid(path, begin, end):
    trim = f'trim_{path[3:]}'
    os.system(f'ffmpeg -y -i {path} -ss {begin} -to {end} -c copy {trim}')
    return trim


def first_frame(path):
    stream = VideoGear(source=path).start()
    frame = Image.fromarray(cvt(stream.read()))
    stream.stop()
    return frame


def prepare(path):
    vid = VideoInfo.from_video_path(path)

    if which('ffmpeg'):
        trimmed = sb.checkbox('Trim')
        if trimmed:
            length = int(vid.total_frames / vid.fps)
            begin, end = sb.slider(
                'Trim by second',
                value=(0, length),
                max_value=length,
            )
            begin, end = hms(begin), hms(end)
            if sb.button(f'Trim from {begin[3:]} to {end[3:]}'):
                path = trim_vid(path, begin, end)
                session_state['path'] = path
        else:
            session_state['path'] = path
    else:
        session_state['path'] = path


def exe_button(text, cmd):
    if sb.button(text):
        st.code(cmd, language='bash')
        os.system(cmd)


def native_run(source):
    cmd = f'{Path(__file__).parent}/native.py --source {source}'

    option = sb.radio(
        f"Native run on {source if source != 0 else 'camera'}",
        ('Show', 'Save to video'),
    )
    if option == 'Show':
        exe_button('Show with OpenCV', cmd)
    elif option == 'Save to video':
        saveto = sb.text_input(' ', 'result.mp4', label_visibility='collapsed')
        exe_button('Save with OpenCV', f'{cmd} --saveto {saveto}')


def main(state):
    st_config()

    info = load()
    model = Model(info)

    def predict_image(file):
        f = np.array(Image.open(file))
        if model.legacy:
            det = Detections.from_yolov5(model.model(f))
            f = BoxAnnotator().annotate(
                scene=f,
                detections=det,
                labels=[
                    f'{conf:0.2f} {model.names[cls]}' for _, _, conf, cls, _ in det
                ],
            )
        else:
            f = model(f)[0].plot()
        st.image(f)

    def predict_video(source, bg, reso):
        width, height = reso
        task = model.info.task
        cam_open = st.checkbox('Run & show on web')

        an = (
            Annotator.ui(info, reso, bg)
            if task != 'classify'
            else Annotator(model=model, reso=reso)
        )

        cap = cv2.VideoCapture(source)
        codec = cv2.VideoWriter_fourcc(*'MJPG')
        cap.set(6, codec)
        cap.set(5, 30)
        cap.set(3, width)
        cap.set(4, height)

        with st.empty():
            while cam_open:
                _, f = cap.read()
                sv_out, yolo_out = an(f)
                t1, t2 = st.tabs(['Supervision', 'YOLO'])
                t1.image(cvt(sv_out))
                t2.image(yolo_out)
        cap.release()

        if sb.button('Save config to json'):
            an.dump('config.json')
        native_run(source)

        if source == 0:

            def cam_stream(key, callback):
                webrtc_streamer(
                    key=key,
                    video_frame_callback=callback,
                    media_stream_constraints={
                        'video': {
                            'width': {'min': width},
                            'height': {'min': height},
                        }
                    },
                )

            def simplecam(frame):
                f = cvt(model(frame.to_ndarray(format='bgr24'))[0].plot())
                return VideoFrame.from_ndarray(f)

            # oh my god, it took me so long to realize the frame bigger through time
            def cam(frame):
                f = frame.to_ndarray(format='bgr24')
                global _shape
                if f.shape != _shape:
                    _shape = f.shape
                    an.update(f)
                return VideoFrame.from_ndarray(cvt(an(f)[0]))

            if task != 'classify':
                cam_stream('b', cam)
            else:
                cam_stream('a', simplecam)

    file = sb.file_uploader(' ', label_visibility='collapsed')
    bg = None

    if sb.checkbox('Use Camera'):
        file = None
        reso = maxcam()

        if sb.checkbox('Annotate from selfie'):
            bg = st.camera_input('Shoot')
        if bg:
            predict_image(bg)
            bg = Image.open(bg).resize(reso)

        predict_video(0, bg, reso)

    if file:
        if 'image' in file.type:
            sb.image(file)
            predict_image(file)

        elif 'video' in file.type:
            sb.video(file)
            path = f'up_{file.name}'
            with open(path, 'wb') as up:
                up.write(file.read())

            prepare(path)
            path = session_state['path']
            vid = VideoInfo.from_video_path(path)
            reso = vid.resolution_wh
            sb.markdown(
                f'''
            - Video resolution: {'x'.join([str(i) for i in reso])}
            - Total frames: {vid.total_frames}
            - FPS: {vid.fps}
            - Path: {path}
                '''
            )
            predict_video(path, first_frame(path), reso)
        else:
            sb.warning('Please upload image/video')


class App(LightningFlow):
    def configure_layout(self):
        return StreamlitFrontend(render_fn=main)

    def run(self):
        pass


lit = LightningApp(App())

running_apps = [i for i in [p.cmdline() for p in process_iter()] if 'run' in i]
this_process = next(p for p in running_apps if any(Path(__file__).stem in a for a in p))

if 'app' not in this_process:
    main('')
