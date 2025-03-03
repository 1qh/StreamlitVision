#!/usr/bin/env python3
from cv2 import destroyAllWindows, imshow, waitKey
from typer import run
from vidgear.gears import WriteGear

from core import Annotator


def app(source: str | int = 0, config: str = 'config.json', output: str | None = None) -> None:
  if '.' not in str(source) and int(source) in range(-1, 2):
    source = int(source)

  gen = Annotator.load(config).gen(source)

  if output is None:
    for f in gen:
      imshow('', f)
      if waitKey(1) & 0xFF == ord('q'):
        break
    destroyAllWindows()
  else:
    writer = WriteGear(output=output)
    for f in gen:
      writer.write(f)
    writer.close()


if __name__ == '__main__':
  run(app)
