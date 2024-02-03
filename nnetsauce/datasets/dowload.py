import pandas as pd
import requests


class Downloader:

    def __init__(self):
        self.pkgname = None
        self.dataset = None
        self.source = None
        self.url = None
        self.request = None    

    def download(self,
        pkgname="MASS",
        dataset="Boston",
        source="https://cran.r-universe.dev/",
        **kwargs):
        self.pkgname = pkgname
        self.dataset = dataset
        self.source = source
        self.url = source + pkgname + "/data/" + dataset + "/json"
        self.request = requests.get(self.url)
        return pd.DataFrame(self.request.json(), **kwargs)
