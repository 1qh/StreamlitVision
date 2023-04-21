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

- Run on LightningAI/Streamlit

  - Model

    - Object detection
    - Object segmentation
    - Pose estimation
    - Image classification

  - on

    - image
    - video
    - webcam

  - with ability to

    - turn tracking on/off
    - adjust confidence threshold
    - filter by class
    - trim video

- Draw visual elements interactively

  - line count (in/out)
  - polygon zone count

- Adjust visual elements

  - text size
  - text color
  - text padding
  - text offset
  - line thickness
  - mask opacity

- Run in native OpenCV with saved visual elements configuration

<details><summary>Note</summary>

### Limitations

- Everything tested on Linux. Some might not be compatible with Windows/Mac.

- Currently only YOLOv8 models are supported.

</details>
