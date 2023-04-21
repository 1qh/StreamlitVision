# Computer Vision Web UI

If you're running on CPU, make sure you run this first:

```
pip install -i https://download.pytorch.org/whl/cpu torch torchvision
```

### Setup

```
pip install -r requirements.txt
pip install lap
```

### Run

| LightningAI                | Streamlit              |
| -------------------------- | ---------------------- |
| `lightning run app app.py` | `streamlit run app.py` |

# Features

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
    - Trim video

- Draw visual elements interactively

  - Line count (in/out)
  - Polygon zone count

- Customize visual elements

  - Toggle on/off

    - Box
    - Mask
    - Area

  - Adjust

    - Text size
    - Text color
    - Text padding
    - Text offset
    - Line thickness
    - Mask opacity

- Run in native OpenCV with saved visual elements configuration

<details><summary>Note</summary>

### Limitations

- Everything tested on Linux. Some might not be compatible with Windows/Mac.

- Currently only YOLOv8 models are supported.

</details>
