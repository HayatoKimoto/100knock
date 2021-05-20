import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

x_train = pd.read_table('ans50/train.feature.tsv', header=None)
y_train = pd.read_table('ans50/train.tsv', header=None)[1]

clf = LogisticRegression(penalty='l2', solver='sag', random_state=0)
clf.fit(x_train, y_train)
joblib.dump(clf, 'ans50/model.joblib')