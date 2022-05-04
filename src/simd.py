
import collections
import jpeglib
import numpy as np
from pathlib import Path
from scipy.stats import ttest_ind, ttest_ind_from_stats
import tempfile
from typing import Tuple

TestResults = collections.namedtuple('TestResults', ['compression','decompression'])

def is_turbo_faster_than_6b(dataset: np.ndarray) -> Tuple[float,float]:
    """"""
    test_versions = ['6b','turbo210']
    with tempfile.TemporaryDirectory() as tmp:
        fnames = [str(Path(tmp) / f'{i}.jpeg') for i in range(dataset.shape[0])]
        # timing of compression and decompression
        t_c,t_d = {v:[] for v in test_versions},{v:[] for v in test_versions} # compression and decompression times
        for v in test_versions:
            jpeglib.version.set(v)
            # iterate dataset
            for i in range(dataset.shape[0]):
                t = jpeglib.Timer('compression')
                im = jpeglib.from_spatial(dataset[i])
                im.samp_factor = ((2,2),(1,1),(1,1))
                im.write_spatial(fnames[i])
                t_c[v].append(t.stop())
                
                t = jpeglib.Timer('decompression')
                jpeglib.read_spatial(fnames[i]).spatial
                t_d[v].append(t.stop())

    # Welch's t-test
    #   H0: mu_6b <= mu_turbo
    #   HA: mu_6b > mu_turbo
    _, pc = ttest_ind(t_c['6b'], t_c['turbo210'], alternative='greater', equal_var=False)
    _, pd = ttest_ind(t_d['6b'], t_d['turbo210'], alternative='greater', equal_var=False)
    return TestResults(pc, pd)
    print('p-values: for compression %.2E, for decompression %.2E' % (pc,pd))

# # plot histograms
# import pandas as pd
# times_c = pd.concat([pd.DataFrame({'library': 'jpeglib-turbo', 'time': t_c['turbo210']}),
#                      pd.DataFrame({'library': 'jpeglib 6b', 'time': t_c['6b']}) ], ignore_index=True)
# times_d = pd.concat([pd.DataFrame({'library': 'jpeglib-turbo', 'time': t_d['turbo210']}),
#                      pd.DataFrame({'library': 'jpeglib 6b', 'time': t_d['6b']}) ], ignore_index=True)
# import seaborn as sns
# fig,ax = plt.subplots(1,2, figsize=(15,5))
# sns.kdeplot(data=times_c, x="time", hue="library", ax=ax[0]).set_title("Compression time for 256x256 colored image");
# sns.kdeplot(data=times_d, x="time", hue="library", ax=ax[1]).set_title("Decompression time for 256x256 colored image");
# ax[0].set_xlim(0,.005)
# ax[1].set_xlim(0,.005)
# plt.show()

