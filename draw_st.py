import pandas as pd
import streamlit as st
from PIL import Image
from streamlit import set_page_config
from streamlit import sidebar as sb
from streamlit_drawable_canvas import st_canvas

set_page_config(layout='wide')

mode = sb.selectbox(
    'Mode', ('freedraw', 'line', 'rect', 'circle', 'transform', 'polygon', 'point')
)
stroke_width = sb.slider('Stroke width', 0, 25, 3)
transparency = sb.slider('Fill transparency', 0x00, 0xFF, 0x99)
fill_color = sb.color_picker('Fill color', '#123456') + hex(transparency)[2:].zfill(2)
stroke_color = sb.color_picker('Stroke color')
realtime = sb.checkbox('Realtime update', True)


def mycanvas(**kwargs):
    return st_canvas(
        fill_color=fill_color,
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        update_streamlit=realtime,
        drawing_mode=mode,
        point_display_radius=sb.slider('Point display radius', 1, 25, 3)
        if mode == 'point'
        else 0,
        key='a',
        **kwargs
    )


if sb.checkbox('Background as image'):
    bg = sb.file_uploader('Upload', type=['png', 'jpg', 'jpeg'])
    if bg:
        bg = Image.open(bg)
        canvas = mycanvas(
            background_image=bg,
            height=bg.size[1],
            width=bg.size[0],
        )
        if canvas.image_data is not None:
            st.image(
                Image.alpha_composite(
                    bg.convert('RGBA'),
                    Image.fromarray(canvas.image_data),
                )
            )
else:
    bg_color = sb.color_picker('Background color', '#eeeeee')
    height = sb.number_input('Height', value=720)
    width = sb.number_input('Width', value=1280)
    if sb.checkbox('Show'):
        canvas = mycanvas(
            background_color=bg_color,
            height=height,
            width=width,
        )
        if canvas.image_data is not None:
            st.image(canvas.image_data)
        if canvas.json_data is not None:
            obj = pd.json_normalize(canvas.json_data['objects'])
            for c in obj.select_dtypes(include=['object']).columns:
                obj[c] = obj[c].astype('str')
            st.dataframe(obj)
