import pandas as pd
import numpy as np
import sklearn
from sklearn.mixture import GaussianMixture


class Main:
    if __name__ == '__main__':
        # We will calculate a similarity matrix similar to the first assignment.
        print("Hello World")
        data = pd.read_csv('data/features_no_txt.csv')
        data = data.drop(columns=['Filename'])
        train, development, test = np.split(data.sample(frac=1), [int(.6 * len(data)), int(.8 * len(data))])
        # First find out the optimum number of components using the development set.
        development_target = development[['Label']]
        development_given = development.drop(columns=['Label'])
        componentCount = 5
        isModelImproving = True
        # while isModelImproving:
#        gmm = GaussianMixture(n_components=componentCount, covariance_type='full', random_state=42)

        gmm = GaussianMixture()

        gmm.fit(X=development_given, y=development_target)

        gmm = gmm.bic(X=development_given)

        target = data[['Label']]
        features = data.drop(columns=['Label'])
        # print(data)
