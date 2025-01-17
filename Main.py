import copy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
from operator import itemgetter
import sklearn
from matplotlib.pyplot import plot
from pandas import DataFrame
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

from FusionedModel import FusionedModel


def shuffle_split_data(subjects):
    subjectsSplit = [[None] * len(subjects)] * 3
    for i in range(len(subjects)):
        train, development, test \
            = np.split(subjects[i].sample(frac=1), [int(.6 * len(subjects[i])), int(.8 * len(subjects[i]))])
        # frac = 1 ensures that the data set is shuffled
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

def calculate_hter(matrix, thresholds):
    far = [None] * len(matrix)
    frr = [None] * len(matrix)
    approximation_steps = 0
    acceptedAndGen = [0] * len(matrix)
    acceptedAndImpo = [0] * len(matrix)
    rejectedAndGen = [0] * len(matrix)
    rejectedAndImpo = [0] * len(matrix)
    for s in range(len(matrix)):
        for r in range(len(matrix[s])):
            for m in range(len(matrix[s][r])):
                if matrix[s][r][m] is not None:
                    if matrix[s][r][m] < thresholds[s]:
                        if s == m:
                            rejectedAndGen[s] += 1
                        else:
                            rejectedAndImpo[s] += 1
                    else:
                        if s == m:
                            acceptedAndGen[s] += 1
                        else:
                            acceptedAndImpo[s] += 1
        far[s] = acceptedAndImpo[s] / (acceptedAndImpo[s] + rejectedAndImpo[s])
        frr[s] = rejectedAndGen[s] / (rejectedAndGen[s] + acceptedAndGen[s])
    total_far = sum(acceptedAndImpo) / (sum(acceptedAndImpo)+sum(rejectedAndImpo))
    total_frr = sum(rejectedAndGen) / (sum(rejectedAndGen) + sum(acceptedAndGen))
    if ((total_frr+total_far)/2)>0.3:
        print('stop')
    return (total_frr+total_far)/2


def calculate_user_dependent_eer(matrix):
    lower_bounds = [0] * len(matrix)
    upper_bounds = [1] * len(matrix)
    thresholds = [0.5] * len(matrix)
    far = [None] * len(matrix)
    frr = [None] * len(matrix)
    approximation_steps = 0
    while approximation_steps < 30:
        acceptedAndGen = [0] * len(matrix)
        acceptedAndImpo = [0] * len(matrix)
        rejectedAndGen = [0] * len(matrix)
        rejectedAndImpo = [0] * len(matrix)
        for s in range(len(matrix)):
            for r in range(len(matrix[s])):
                for m in range(len(matrix[s][r])):
                    if matrix[s][r][m] is not None:
                        if matrix[s][r][m] < thresholds[s]:
                            if s == m:
                                rejectedAndGen[s] += 1
                            else:
                                rejectedAndImpo[s] += 1
                        else:
                            if s == m:
                                acceptedAndGen[s] += 1
                            else:
                                acceptedAndImpo[s] += 1
            far[s] = acceptedAndImpo[s] / (acceptedAndImpo[s] + rejectedAndImpo[s])
            frr[s] = rejectedAndGen[s] / (rejectedAndGen[s] + acceptedAndGen[s])
            if far[s] > frr[s]:
                lower_bounds[s] = thresholds[s]
                thresholds[s] = (upper_bounds[s]+lower_bounds[s])/2
            elif far[s] < frr[s]:
                upper_bounds[s] = thresholds[s]
                thresholds[s] = (upper_bounds[s]+lower_bounds[s])/2
        approximation_steps += 1
    total_far = sum(acceptedAndImpo) / (sum(acceptedAndImpo)+sum(rejectedAndImpo))
    total_frr = sum(rejectedAndGen) / (sum(rejectedAndGen) + sum(acceptedAndGen))
    return (total_frr+total_far)/2, thresholds

def gen_impo_histogram(matrix):
    gen_scores = []
    impo_scores = []
    for s in range(len(matrix)):
        for r in range(len(matrix[s])):
            for m in range(len(matrix[s][r])):
                if matrix[s][r][m] is not None:
                    if s == m:
                        gen_scores.append(matrix[s][r][m])
                    else:
                        impo_scores.append(matrix[s][r][m])
    plt.hist(gen_scores,5)
    plt.hist(impo_scores,150)

    plt.savefig('./data/hist.png')


def plot_far_frr_curve(matrix):
    far = [None]*100
    frr = [None]*100
    for titerator in range(0,100):
        t_hold = titerator / 100
        acceptedAndGen = 0
        acceptedAndImpo = 0
        rejectedAndGen = 0
        rejectedAndImpo = 0
        for s in range(len(matrix)):
            for r in range(len(matrix[s])):
                for m in range(len(matrix[s][r])):
                    if matrix[s][r][m] is not None:
                        if matrix[s][r][m] < t_hold:
                            if s == m:
                                rejectedAndGen += 1
                            else:
                                rejectedAndImpo += 1
                        if matrix[s][r][m] >= t_hold:
                            if s == m:
                                acceptedAndGen += 1
                            else:
                                acceptedAndImpo += 1

        far[titerator] = acceptedAndImpo / (acceptedAndImpo + rejectedAndImpo)
        frr[titerator] = rejectedAndGen / (rejectedAndGen + acceptedAndGen)

    plt.plot(far)
    plt.plot(frr)
    plt.savefig('./data/far_frr_curve.png')


def calculate_EER_on_normalized_matrix(matrix):
    upper_bound = 1
    lower_bound = 0
    while (upper_bound - lower_bound) > 0.001:
        t_hold = (upper_bound + lower_bound) / 2
        acceptedAndGen = 0
        acceptedAndImpo = 0
        rejectedAndGen = 0
        rejectedAndImpo = 0
        for s in range(len(matrix)):
            for r in range(len(matrix[s])):
                for m in range(len(matrix[s][r])):
                    if matrix[s][r][m] is not None:
                        if matrix[s][r][m] < t_hold:
                            if s == m:
                                rejectedAndGen += 1
                            else:
                                rejectedAndImpo += 1
                        if matrix[s][r][m] >= t_hold:
                            if s == m:
                                acceptedAndGen += 1
                            else:
                                acceptedAndImpo += 1

        far = acceptedAndImpo / (acceptedAndImpo + rejectedAndImpo)
        frr = rejectedAndGen / (rejectedAndGen + acceptedAndGen)
        if far > frr:
            lower_bound = t_hold
        elif far < frr:
            upper_bound = t_hold
        else:
            break
    return (far+frr)/2, t_hold

def calculate_sim_matrix(models, developmentSet):
    max_row = 0
    for data_frame in developmentSet:
        if max_row < len(data_frame):
            max_row = len(data_frame)

    # sim_matrix = [[[None] * len(models)] * max_row] * len(developmentSet)
    sim_matrix = [[[None for k in range(len(models))] for j in range(max_row)] for i in range(len(developmentSet))]
    for m in range(len(models)):
        for s in range(len(developmentSet)):
            per_sample_similarities = models[m].score_samples(developmentSet[s])
            for r in range(len(developmentSet[s])):
                sim_matrix[s][r][m] = per_sample_similarities[r]
    return sim_matrix

def calculate_normalized_similarity_matrix(models, developmentSet):
    sim_matrix = calculate_sim_matrix(models, developmentSet)
    minimum_entry = find_min(sim_matrix)
    sim_matrix = operation_on_each_element(sim_matrix, lambda a: a - minimum_entry)
    maximum_entry = find_max(sim_matrix)
    sim_matrix = operation_on_each_element(sim_matrix, lambda a: a / maximum_entry)
    return sim_matrix

def concat_background_data(user_index, user_divided_data):
    concated_data = pd.DataFrame()
    for i in range(len(user_divided_data)):
        if i != user_index:
            concated_data = concated_data.append(user_divided_data)
    return concated_data

def merge_background_data(user_divided_data):
    background_data = [None] * len(user_divided_data)
    for user_index in range(len(background_data)):
        background_data[user_index] = concat_background_data(user_index, user_divided_data)
    return background_data

def extract_subjects():
    # We will calculate a similarity matrix similar to the first assignment.
    data = pd.read_csv('data/features_no_txt.csv')
    data = data.drop(columns=['Filename'])
    dataGrouped = data.groupby('Label')
    users = data['Label'].unique()
    subjects = [DataFrame()] * 10
    lastIndex = 0
    for user in users:
        subjects[lastIndex] = data.loc[data['Label'] == user]
        lastIndex = lastIndex + 1
    # The array 'subjects' now contains the rows for each subject in separate fields

    # Drop the Labels since they are now identified by their indices.
    for index in range(len(copy.deepcopy(subjects))):
        subjects[index] = subjects[index].drop(columns=['Label'])
    return subjects


def hter_with_thold(best_models, test_set, best_thresholds): # testset contains the test data for each subject [0..9]
    return calculate_hter(calculate_normalized_similarity_matrix(best_models,test_set), best_thresholds)

# the subjects contain all the data for each subject in the array.
def second_section():
    # subjectsSplit: for each in (train, develop, test) -> for each user -> subset of data for that user and for that purpose.

    eer_development_set_user_dependent = [None] * 5
    hter_test_set_user_dependent = [None] * 5

    subjects = extract_subjects()
    best_foreground_models = [[None]*len(subjects)]*5
    ubm_models = [[None]*len(subjects)]*5
    merged_models = [[None]*len(subjects)]*5
    n_components_ubm = 1
    best_n_components_ubm = 1
    best_EER = 1
    best_models = [[None]*len(subjects)]*5
    eer_increased_n_times = 0
    best_thresholds = [None]*5  # In this list the user specific thresholds are written for each fold. (2d arr actually)
    hter_test_set_onle_foreground = [None] *5
    for experiment_count in range(5):
        subjectsSplit = shuffle_split_data(subjects)
        background_data_train = merge_background_data(subjectsSplit[0])

        for user_index in range(len(subjects)):
            best_foreground_models[experiment_count][user_index] = GaussianMixture(n_components=28)
            best_foreground_models[experiment_count][user_index].fit(subjectsSplit[0][user_index])

        test_current_eer, test_thresholds = calculate_user_dependent_eer(calculate_normalized_similarity_matrix(best_foreground_models[experiment_count],subjectsSplit[1]))
        print('EER:', test_current_eer, 'value of foreground model, no of components: ', 28, 'fold',experiment_count)

        while eer_increased_n_times <3:
            for user_index in range(len(subjects)):
                # train the ubm with the user specific noise
                ubm_models[experiment_count][user_index] = GaussianMixture(n_components=n_components_ubm)
                ubm_models[experiment_count][user_index].fit(background_data_train[user_index])
                # train hte foreground model with user data

                merged_models[experiment_count][user_index] = FusionedModel(best_foreground_models[experiment_count][user_index],ubm_models[experiment_count][user_index])

            current_eer, thresholds = calculate_user_dependent_eer(calculate_normalized_similarity_matrix(merged_models[experiment_count],
                                                                                   subjectsSplit[1]))
            if best_EER > current_eer:
                best_EER = current_eer
                eer_development_set_user_dependent[experiment_count] = current_eer
                best_thresholds[experiment_count] = thresholds
                best_models = copy.deepcopy(merged_models[experiment_count])
                best_n_components_ubm = n_components_ubm
                eer_increased_n_times = 0
            else:
                eer_increased_n_times += 1
            print('EER:', current_eer, ' components: ', n_components_ubm)
            n_components_ubm += 1

        plot_far_frr_curve(calculate_normalized_similarity_matrix(best_models, subjectsSplit[2]))
        gen_impo_histogram(calculate_normalized_similarity_matrix(best_models, subjectsSplit[2]))

        hter_test_set_user_dependent[experiment_count] \
            = hter_with_thold(merged_models[experiment_count], subjectsSplit[2], best_thresholds[experiment_count])
        hter_test_set_onle_foreground[experiment_count] \
            = hter_with_thold(best_foreground_models[experiment_count], subjectsSplit[2], best_thresholds[experiment_count])
        n_components_ubm = 1
        best_EER=1
        eer_increased_n_times=0
    print('hter on test set: ', hter_test_set_user_dependent)
    print('hter on test set only foreground: ', hter_test_set_onle_foreground)
    print('eer on development set: ', eer_development_set_user_dependent)
    print('n of components of background model: ', list(map(lambda x: x[0].get_n_components_of_background(), merged_models)))




def find_optimum_components(subjectsSplit, subjects, thresholding_method):
    # Calculate the number of GMM models for each subject independently and save them in an array.
    models = [None] * len(subjects)
    best_component_number = 1
    best_EER = 1
    best_models = [None] * len(subjects)
    best_component_number = 1
    eer_increased_n_times = 0
    componentNumber = 0
    best_threshold = -1

    # Even though we may not find the global minimum eer, some experiments have shown that
    # we are reaching a good eer with this approach.
    # The EER is fluctuating alot for a very high number of components.
    while eer_increased_n_times < 3:
        componentNumber += 1
        for index in range(len(subjects)):
            gmm = GaussianMixture(n_components=componentNumber, n_init=1)
            gmm.fit(subjectsSplit[0][index])
            models[index] = gmm
        current_eer, thold_result = thresholding_method(calculate_normalized_similarity_matrix(models, subjectsSplit[1]))

        if best_EER > current_eer:
            best_threshold = thold_result
            best_EER = current_eer
            best_models = copy.deepcopy(models)
            best_component_number = componentNumber
            eer_increased_n_times = 0
        else:
            eer_increased_n_times += 1
        print('EER:', current_eer, ' components: ',
              componentNumber)
    print('Optimum no of components for the foreground model: ', best_component_number)
    return best_EER, best_models, best_threshold


class Main: #first section
    if __name__ == '__main__':
        subjects = extract_subjects()
        # The rows for each subject will be divided into three sets for training development and test.
        eer_development_set_user_independent = [None]*5
        eer_development_set_user_dependent = [None]*5

        thold_best_eer_fold = [None]*5
        thold_best_eer_fold_user_dependent = [[None]*len(subjects)]*5

        models_user_independent = [[None]*len(subjects)]*5
        models_user_dependent = [[None]*len(subjects)]*5

        hter_test_set_user_dependent = [None] * 5
        hter_test_set_user_independent = [None]*5


        for experiment_count in range(5):
            subjectsSplit = shuffle_split_data(subjects)
            print('Fold no: ', experiment_count)


            eer_development_set_user_dependent[experiment_count], models_user_dependent[experiment_count], tholds_user_depend \
                = find_optimum_components(subjectsSplit,
                                          subjects,
                                          calculate_user_dependent_eer)
            print('eer_development_set_user_dependent', eer_development_set_user_dependent[experiment_count])

            plot_far_frr_curve(calculate_normalized_similarity_matrix(models_user_dependent[experiment_count], subjectsSplit[2]))
            gen_impo_histogram(calculate_normalized_similarity_matrix(models_user_dependent[experiment_count], subjectsSplit[2]))


            hter_test_set_user_dependent[experiment_count] \
                = hter_with_thold(models_user_dependent[experiment_count], subjectsSplit[2], tholds_user_depend)
            print('hter_test_set_user_dependent', hter_test_set_user_dependent[experiment_count])


            eer_development_set_user_independent[experiment_count], models_user_independent[experiment_count], thold_user_inde \
                = find_optimum_components(subjectsSplit,
                                          subjects,
                                          calculate_EER_on_normalized_matrix)
            print('eer_development_set_user_independent', eer_development_set_user_independent[experiment_count])

            hter_test_set_user_independent[experiment_count] \
                = hter_with_thold(models_user_independent[experiment_count],subjectsSplit[2],([thold_user_inde]*len(subjects)))
            print('hter_test_set_user_independent', hter_test_set_user_independent[experiment_count])



        print('eer_development_set_user_dependent',eer_development_set_user_dependent)
        print('models_user_dependent', list(map(lambda x: x[0].n_components, models_user_dependent)))
        print('hter_test_set_user_dependent',hter_test_set_user_dependent)

        print('eer_development_set_user_independent',eer_development_set_user_independent)
        print('models_user_independent', list(map(lambda x: x[0].n_components, models_user_independent)))
        print('hter_test_set_user_independent',hter_test_set_user_independent)

        print('Average eer development set user dependent: ', (sum(eer_development_set_user_dependent)/5),
              'Standard deviation: ', np.std(eer_development_set_user_dependent))
        print('Average eer development set user independent: ', (sum(eer_development_set_user_independent)/5),
              'Standard deviation: ', np.std(eer_development_set_user_independent))
        print('Average hter test set user dependent: ', (sum(hter_test_set_user_dependent)/5),
              'Standard deviation: ', np.std(hter_test_set_user_dependent))
        print('Average hter test set user independent: ', (sum(hter_test_set_user_independent)/5),
              'Standard deviation: ', np.std(hter_test_set_user_independent))

        # Pick the user dependent models as our best models to do the second section of the homework.

        index_of_min = min(enumerate(hter_test_set_user_dependent), key=itemgetter(1))[0]
        print('Use the following no of components: ', models_user_dependent[index_of_min][8].n_components)
        second_section()

    #second_section(best_models, subjects, subjectsSplit)

#       According to sklearn documentation, the GMM here already uses the Mahalanobis distance by default
#       https://scikit-learn.org/stable/modules/clustering.html
