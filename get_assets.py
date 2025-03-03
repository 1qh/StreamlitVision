from os import chdir
from pathlib import Path
from subprocess import DEVNULL, check_output, run

from supervision.assets import VideoAssets, download_assets


def get_size(path: str) -> str:
  return (
    check_output(
      f'ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 {path}',
      shell=True,
    )
    .decode()
    .strip()
  )


def half_downscale(original_folder: str, new_folder: str) -> None:
  Path(new_folder).mkdir(parents=True, exist_ok=True)
  for video in Path(original_folder).iterdir():
    if video.suffix == '.mp4':
      run(
        f'ffmpeg -y -i {original_folder}/{video} -vf scale="iw/2:ih/2" {new_folder}/{video}',
        shell=True,
        stderr=DEVNULL,
        stdin=DEVNULL,
        stdout=DEVNULL,
        check=False,
      )
      original = get_size(f'{original_folder}/{video}')
      modified = get_size(f'{new_folder}/{video}')
      print(f'{video}: {original} -> {modified}')


Path('videos').mkdir(parents=True, exist_ok=True)
chdir('videos')
for asset in VideoAssets.list():
  download_assets(asset)

chdir('..')
half_downscale('videos', 'half_videos')
half_downscale('half_videos', 'quarter_videos')
