import os
from shutil import which
from time import gmtime, strftime

import cv2
import streamlit as st
from PIL import Image
from streamlit import set_page_config
from streamlit import sidebar as sb
from ultralytics import YOLO

set_page_config(layout='wide')


def custom_classes(model):
    d = model.names
    all = list(d.values())

    if sb.checkbox('Custom Classes'):
        return [all.index(i) for i in sb.multiselect(' ', all)]
    else:
        return list(d.keys())


def dis(im):
    st.image(cv2.cvtColor(im.plot(), cv2.COLOR_BGR2RGB))


def hms(s):
    return strftime('%H:%M:%S', gmtime(s))


conf = sb.slider('Threshold', max_value=1.0, value=0.25)
m = YOLO('yolov8n-seg.pt') if sb.checkbox('Segment') else YOLO('yolov8n.pt')
classes = custom_classes(m)
file = sb.file_uploader(' ')

if file:
    if 'image' in file.type:
        sb.image(file)
        s = Image.open(file)
        dis(
            m(
                s,
                classes=classes,
                conf=conf,
                retina_masks=True,
            )[0]
        )

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
                    value=(
                        0,
                        length,
                    ),
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
                for i in m(
                    s,
                    stream=True,
                    classes=classes,
                    conf=conf,
                    retina_masks=True,
                ):
                    dis(i)
