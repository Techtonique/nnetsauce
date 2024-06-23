import numpy as np
import os
import unittest as ut
import nnetsauce as ns

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")


class TestDownload(ut.TestCase):
    def test_download(self):
        # the controversial Boston dataset
        df1 = ns.Downloader().download(dataset="Boston")
        print(f"df1.iloc[0, 0]: {df1.iloc[0, 0]}")
        self.assertTrue(np.allclose(df1.iloc[0, 0], 0.0063))


if __name__ == "__main__":
    ut.main()
