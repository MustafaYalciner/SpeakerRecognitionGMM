import copy
import pandas as pd
import numpy as np
import sklearn
from pandas import DataFrame
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt


def shuffle_split_data(subjects):
    subjectsSplit = [[None]*len(subjects)]*3
    for i in range(len(subjects)):
        train, development, test \
            = np.split(subjects[i].sample(frac=1), [int(.6 * len(subjects[i])), int(.8 * len(subjects[i]))])
        subjectsSplit[0][i] = train
        subjectsSplit[1][i] = development
        subjectsSplit[2][i] = test
    return subjectsSplit

def find_min(matrix):
    min = matrix[0][0][0]
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            for k in range(len(matrix[i][j])):
                if (matrix[i][j][k] is not None) and (matrix[i][j][k] < min):
                    min = matrix[i][j][k]
    return min

def find_max(matrix):
    max = matrix[0][0][0]
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            for k in range(len(matrix[i][j])):
                if (matrix[i][j][k] is not None) and (matrix[i][j][k] > max):
                    max = matrix[i][j][k]
    return max

def operation_on_each_element(matrix, lmd):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            for k in range(len(matrix[i][j])):
                if matrix[i][j][k] is not None:
                    matrix[i][j][k] = lmd(matrix[i][j][k])
    return matrix

def calculate_EER(models, developmentSet):
    sim_matrix = calculate_sim_matrix(models, developmentSet)
    minimum_entry = find_min(sim_matrix)
    sim_matrix = operation_on_each_element(sim_matrix,lambda a : a-minimum_entry)
    maximum_entry = find_max(sim_matrix)
    sim_matrix = operation_on_each_element(sim_matrix, lambda a : a/maximum_entry)
    return sim_matrix

def calculate_sim_matrix(models, developmentSet):
    max_row = 0
    for data_frame in developmentSet:
        if max_row < len(data_frame):
            max_row = len(data_frame)

    sim_matrix = [[[None]*len(models)]*max_row]*len(developmentSet)
    sim_matrix = [[[None for k in range(len(models))] for j in range(max_row)] for i in range(len(developmentSet))]
    for m in range(len(models)):
        for s in range(len(developmentSet)):
            per_sample_similarities = models[m].score_samples(developmentSet[s])
            for r in range(len(developmentSet[s])):
                sim_matrix[s][r][m] = per_sample_similarities[r]
    return sim_matrix


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
            lastIndex = lastIndex+1
        # The array 'subjects' now contains the rows for each subject in separate fields

        # Drop the Labels since they are now identified by their indices.
        for index in range(len(copy.deepcopy(subjects))):
            subjects[index] = subjects[index].drop(columns=['Label'])

        # The rows for each subject will be divided into three sets for training development and test.
        for experiment_count in range(5):
            subjectsSplit = shuffle_split_data(subjects)
            # Calculate the number of GMM models for each subject independently and save them in an array.
            models = [None] * len(subjects)
            best_component_number = 1
            best_EER = 1
            is_eer_decreasing = True
            componentNumber = 1
            while is_eer_decreasing:
                for index in range(len(subjects)):
                    gmm = GaussianMixture(n_components=componentNumber)
                    gmm.fit(subjectsSplit[0][index])
                    models[index] = gmm
                componentNumber += 1
                eer = calculate_EER(models, subjectsSplit[1])
                is_eer_decreasing = False

            #Calculate the similarity matrix
            print('Number of components for user - is -')
            print(index)
            print(componentNumber)

            print('separator')
#       According to sklearn documentation, the GMM here already uses the Mahalanobis distance by default
#       https://scikit-learn.org/stable/modules/clustering.html



