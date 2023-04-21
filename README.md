If you're running on CPU, make sure you run this first:

```
pip install -i https://download.pytorch.org/whl/cpu torch torchvision
```

## Setup

```
pip install -r requirements.txt
pip install lap
```

## Run

```
lightning run app app.py
```

or

```
streamlit run app.py
```

All functions tested on Linux. Not all will work on Windows/Mac.
