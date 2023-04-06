import cv2
import streamlit as st
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


conf = sb.slider('Threshold', max_value=1.0, value=0.25)
model = YOLO('yolov8n-seg.pt') if sb.checkbox('Segment') else YOLO('yolov8n.pt')
classes = custom_classes(model)

# THIS ONLY WORKS FOR THE FIRST RUN
# PLEASE HELP IF YOU KNOW HOW TO FIX

while sb.checkbox('Run'):
    with st.empty():
        for i in model(1, classes=classes, conf=conf, retina_masks=True, stream=True):
            st.image(cv2.cvtColor(i.plot(), cv2.COLOR_BGR2RGB))
