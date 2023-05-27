# Computer Vision Web UI

### Setup

<details><summary>CPU Only</summary>
  
Install CPU version of pytorch before other dependencies 
```
pip install -i https://download.pytorch.org/whl/cpu torch torchvision
```

</details>

```
pip install -r requirements.txt
pip install lap
```

### Run

| LightningAI                | Streamlit              |
| -------------------------- | ---------------------- |
| `lightning run app app.py` | `streamlit run app.py` |

## Features

- Run locally on [LightningAI](https://github.com/lightning-ai/lightning) / [Streamlit](https://github.com/streamlit/streamlit)

  - Model

    - Object detection
    - Object segmentation
    - Pose estimation
    - Image classification

  - On

    - Image
    - Video
    - Webcam

  - With ability to

    - Turn tracking on/off
    - Adjust confidence threshold
    - Filter by class
    - Object color classification
    - Trim video

- Draw visual elements interactively

  - Line count (in/out)
  - Polygon zone count

- Customize visual elements

  - Toggle on/off

    - Box
    - Mask
    - Area
    - FPS

  - Adjust

    - Text size
    - Text color
    - Text padding
    - Text offset
    - Line thickness
    - Mask opacity

- **PRODUCTION READY**

  - Save drawed visual elements & settings in JSON
  - Run inference with OpenCV standalone from saved JSON

<details><summary>Note</summary>

#### TODO

Supported models:

- [x] All YOLOv8 models (Detect, Segment, Pose, Classify)
  - [x] With tracking
- [x] YOLOv5 object detection
  - [x] new v5u models
  - [x] original v5 models
- [ ] YOLOv6 object detection
- [ ] YOLO-NAS object detection
- [ ] RT-DETR object detection

</details>
