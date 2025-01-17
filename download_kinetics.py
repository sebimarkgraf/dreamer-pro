import logging
import tarfile
import requests
import tqdm
import json
from pytube import YouTube
from pathlib import Path

from pqdm.threads import pqdm

KINETICS400_URL = "https://storage.googleapis.com/deepmind-media/Datasets/kinetics400.tar.gz"


def get_url(path):
    with open(path) as f:
        data = json.load(f)
    urls = []
    for k in data.keys():
        if data[k]["annotations"]["label"] == "driving car":
            urls.append(data[k]["url"])
    return urls


def _download_url(url, dest_path: Path):
    try:
        video = YouTube(url)
        streams = video.streams.filter(file_extension="mp4")
        for stream in streams:
            if stream.resolution == "360p":
                itag = stream.itag
                break
        video.streams.get_by_itag(itag).download(dest_path)
    except Exception as e:
        print(f"Exception encountered in Download: {e}")
        return


def download(urls, dest_path: Path):
    dest_path.mkdir(exist_ok=True, parents=True)
    pqdm(urls, lambda u: _download_url(u, dest_path=dest_path), n_jobs=8)




def download_dataset(path):
    path.mkdir(parents=True, exist_ok=True)
    logging.info("Downloading Kinetics400 dataset.")
    r = requests.get(KINETICS400_URL, stream=True)
    file = tarfile.open(fileobj=r.raw, mode="r|gz")
    file.extractall(path)
    datapath = path / "kinetics400"
    train_urls = get_url(datapath / "train.json")
    test_urls = get_url(datapath / "test.json")

    download(train_urls, datapath / "train")
    download(test_urls, datapath / "test")
    logging.info("Download finished.")

if __name__ == "__main__":
    save_folder = "background/kinetics400"
    download_dataset(Path(save_folder))
