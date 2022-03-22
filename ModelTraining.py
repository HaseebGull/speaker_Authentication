import _pickle as cPickle
import numpy as np
from scipy.io.wavfile import read
from sklearn.mixture import GaussianMixture as GMM
from FeatureExtraction import extract_features
import warnings

warnings.filterwarnings("ignore")
source = "Taining2/"
dest = "Speaker_Models/"
train_file = "trainning2.txt"
file_paths = open(train_file, 'r')

count = 1
features = np.asarray(())
for path in file_paths:
    path = path.strip()
    print(path)

    sr, audio = read(source + path)
    vector = extract_features(audio, sr)

    if features.size == 0:
        features = vector
    else:
        features = np.vstack((features, vector))

    if count == 5:
        gmm = GMM(n_components=5, covariance_type='diag', n_init=3)
        gmm.fit(features)

        picklefile = path.split("-")[0] + ".gmm"
        cPickle.dump(gmm, open(dest + picklefile, 'wb'))
        print('+ modeling completed for speaker:', picklefile, " with data point = ", features.shape)
        features = np.asarray(())
        count = 0
    count = count + 1
