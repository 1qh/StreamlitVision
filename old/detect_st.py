import os
from shutil import which
from time import gmtime, strftime

import cv2
import streamlit as st
from av import VideoFrame
from PIL import Image
from streamlit import set_page_config
from streamlit import sidebar as sb
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO

set_page_config(layout='wide')


def custom_classes(model):
    d = model.names
    all = list(d.values())

    if sb.checkbox('Custom Classes'):
        return [all.index(i) for i in sb.multiselect(' ', all)]
    else:
        return list(d.keys())


def hms(s):
    return strftime('%H:%M:%S', gmtime(s))


def get_frame(i):
    return cv2.cvtColor(i.plot(), cv2.COLOR_BGR2RGB)


def from_model(s, model):
    return get_frame(model(s)[0])


def infer_image(file, model):
    st.image(from_model(Image.open(file), model))


conf = sb.slider('Threshold', max_value=1.0, value=0.25)
model_map = {
    'Detect': 'n',
    'Segment': 'n-seg',
    'Pose': 'n-pose',
    'Classify': 'n-cls',
}
select = sb.selectbox('Model', model_map.keys())
model = YOLO(f'yolov8{model_map[select]}.pt')

classes = custom_classes(model)


def mymodel(source, stream=False):
    return model(source, classes=classes, conf=conf, retina_masks=True, stream=stream)


def cam_stream(frame):
    return VideoFrame.from_ndarray(
        from_model(frame.to_ndarray(format='bgr24'), mymodel)
    )


if sb.checkbox('Use Camera'):
    picture = st.camera_input('Shoot')
    if picture:
        infer_image(picture, mymodel)
    webrtc_streamer(key='a', video_frame_callback=cam_stream)

file = sb.file_uploader(' ')
if file:
    if 'image' in file.type:
        sb.image(file)
        infer_image(file, mymodel)

    if 'video' in file.type:
        sb.video(file)
        s = f'up_{file.name}'

        with open(s, 'wb') as f:
            f.write(file.read())

        if which('ffmpeg'):
            trimmed = sb.checkbox('Trim')

            if trimmed:
                vid = cv2.VideoCapture(s)
                length = int(
                    vid.get(cv2.CAP_PROP_FRAME_COUNT) / vid.get(cv2.CAP_PROP_FPS)
                )
                begin, end = sb.slider(
                    'Trim by second',
                    value=(0, length),
                    max_value=length,
                )
                begin, end = hms(begin), hms(end)
                sb.write(f'Trim from {begin} to {end}')
                trim = f'trim_{file.name}'

        while sb.checkbox('Run'):
            if trimmed:
                os.system(f'ffmpeg -y -i {s} -ss {begin} -to {end} -c copy {trim}')
                s = trim

            with st.empty():
                for i in mymodel(s, stream=True):
                    st.image(get_frame(i))
