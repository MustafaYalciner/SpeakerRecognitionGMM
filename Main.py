import pandas as pd
import numpy as np
import sklearn
from pandas import DataFrame
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt


class Main:
    if __name__ == '__main__':
        # We will calculate a similarity matrix similar to the first assignment.
        print("Hello World")
        data = pd.read_csv('data/features_no_txt.csv')
        data = data.drop(columns=['Filename'])
        dataGrouped = data.groupby('Label')
        users = data['Label'].unique()
        subjects = [DataFrame()] * 10
        lastIndex = 0
        for user in users:
            subjects[lastIndex] = data.loc[data['Label'] == user]
            print(subjects[lastIndex])
            lastIndex = lastIndex+1
        # The array 'subjects' now contains the rows for each subject in separate fields

        # Drop the Labels since they are now identified by their indices.
        for index in range(len(subjects)):
            subjects[index] = subjects[index].drop(columns=['Label'])

        # The rows for each subject will be divided into three sets for training development and test.
        subjectsSplit = [subjects] * 3
        for i in range(len(subjects)):
            train, development, test \
                = np.split(subjects[i].sample(frac=1), [int(.6 * len(subjects[i])), int(.8 * len(subjects[i]))])
            subjectsSplit[0][i] = train
            subjectsSplit[1][i] = development
            subjectsSplit[2][i] = test
        models = [GaussianMixture()] * len(subjects)
        for index in range(len(subjects)):
            componentNumber = 1
            gmm = GaussianMixture(n_components=componentNumber)
            gmm.fit(subjectsSplit[0][0])
            lastBic = gmm.bic(subjectsSplit[0][0])
            currentBic = lastBic-1
            while lastBic > currentBic:
                componentNumber += 1
                gmm = GaussianMixture(n_components=componentNumber)
                gmm.fit(subjectsSplit[0][0])
                lastBic = currentBic
                currentBic = gmm.bic(subjectsSplit[1][0])
            print('Number of components for user - is -')
            print(index)
            print(componentNumber)






