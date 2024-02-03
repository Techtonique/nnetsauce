import pandas as pd
import requests


class Downloader(self):

    def __init__(self, **kwargs):
        self.pkgname = "MASS"
        self.dataset = "Boston"
        self.source = "https://cran.r-universe.dev/"
        self.kwargs = kwargs

    def download(self,
        pkgname="MASS",
        dataset="Boston",
        source="https://cran.r-universe.dev/",
    ):
        self.url = source + pkgname + "/data/" + dataset + "/json"
        self.request = requests.get(self.url)
        return pd.DataFrame(self.request.json(), self.kwargs)
