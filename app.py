import os
from pathlib import Path
from shutil import which
from time import gmtime, strftime

import cv2
import numpy as np
import streamlit as st
from av import VideoFrame
from lightning import LightningApp, LightningFlow
from lightning.app.frontend import StreamlitFrontend
from PIL import Image
from psutil import process_iter
from streamlit import session_state, set_page_config
from streamlit import sidebar as sb
from streamlit_webrtc import webrtc_streamer
from supervision import BoxAnnotator, Detections, VideoInfo
from vidgear.gears import VideoGear

from core import Annotator, Model, cvt, maxcam

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
    div.stButton button {width: 100%;}
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


def hms(s: int) -> str:
    return strftime('%H:%M:%S', gmtime(s))


def trim_vid(path: str, begin: str, end: str) -> str:
    trim = f'trim_{path[3:]}'
    os.system(f'ffmpeg -y -i {path} -ss {begin} -to {end} -c copy {trim}')
    return trim


def first_frame(path: str) -> Image.Image:
    stream = VideoGear(source=path).start()
    frame = Image.fromarray(cvt(stream.read()))
    stream.stop()
    return frame


def prepare(path: str):
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


def exe_button(place, text: str, cmd: str):
    if place.button(text):
        st.code(cmd, language='bash')
        os.system(cmd)


def native_run(place, source: str | int, an: Annotator):
    cmd = f'{Path(__file__).parent}/native.py --source {source}'
    c1, c2 = place.columns([1, 3])
    option = c2.radio(
        f"Native run on {source if source != 0 else 'camera'}",
        ('Realtime inference', 'Save to video'),
    )
    if option == 'Realtime inference':
        exe_button(c1, 'Show with OpenCV', cmd)
    elif option == 'Save to video':
        saveto = c1.text_input(' ', 'result.mp4', label_visibility='collapsed')
        exe_button(c1, 'Save with OpenCV', f'{cmd} --saveto {saveto}')
    if c1.button('Save config to json'):
        an.dump('config.json')


def main(state):
    st_config()
    running = sb.checkbox('Realtime inference')
    model = Model.ui()

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

    def predict_video(
        source: str | int,
        bg: Image.Image,
        reso: tuple[int, int],
    ):
        width, height = reso
        task = model.info.task

        an = (
            Annotator.ui(model, reso, bg)
            if task != 'classify'
            else Annotator(model=model, reso=reso)
        )
        native_run(st, source, an)

        cap = cv2.VideoCapture(source)
        codec = cv2.VideoWriter_fourcc(*'MJPG')
        cap.set(6, codec)
        cap.set(5, 30)
        cap.set(3, width)
        cap.set(4, height)

        if source != 0:
            count = 0
            total_frames = VideoInfo.from_video_path(source).total_frames

        mt = st.empty()
        while running:
            t1, t2 = mt.tabs(['Main', 'Fallback'])
            if task in ('pose', 'classify'):
                t1, t2 = t2, t1
            success, f = cap.read()
            if success:
                f, res = an(f)
                t1.image(cvt(f))
                t2.image(res)
                if source != 0:
                    count += 1
                    t1.progress(count / total_frames)
            else:
                break
        cap.release()

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
