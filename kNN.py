import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as pp
from sklearn import datasets

TRAINING_SIZE = 0
TRAINING_INSTANCES = []

TESTING_SIZE = 0
TESTING_INSTANCES = []

CLASSES = []

K = 10
CURRENT_K = 0
KNN = 0


def stratify_data(data):
    global TRAINING_SIZE
    global TESTING_SIZE
    global TRAINING_INSTANCES
    global TESTING_INSTANCES
    scaler = pp.MinMaxScaler()
    class_value = np.unique(data[:, 0], return_counts=True)
    col_size = np.size(data[0])

    stratified_data = np.empty((0, np.sum(np.array(class_value[1] / K, np.int64)), col_size), np.ndarray)
    split_class = []
    for class_i in np.nditer(class_value):
        sub_class_value = class_i[0]
        sub_class_condition = data[:, 0] == sub_class_value
        split_class.append(data[sub_class_condition])
    for k_i in range(K):
        sub_class_k = np.empty((0, col_size), np.ndarray)
        for class_index in range(len(split_class)):
            sub_class_size = np.size(split_class[class_index], axis=0)
            size_to_add = int(class_value[1][class_index] / K)
            row_num = np.random.choice(range(sub_class_size), size_to_add)
            sub_class_k = np.append(sub_class_k, np.take(split_class[class_index], row_num, axis=0), axis=0)
            split_class[class_index] = np.delete(split_class[class_index], row_num, axis=0)
        stratified_data = np.append(stratified_data, [sub_class_k], axis=0)

    TESTING_SIZE = np.size(stratified_data[0], axis=0)
    TRAINING_SIZE = TESTING_SIZE * (K - 1)
    TRAINING_INSTANCES = np.empty((0, TRAINING_SIZE, col_size), np.ndarray)
    TESTING_INSTANCES = np.empty((0, TESTING_SIZE, col_size), np.ndarray)
    for k_i in range(K):
        testing_data = stratified_data[k_i]
        training_data = np.reshape(np.delete(stratified_data, k_i, axis=0), (TRAINING_SIZE, col_size))
        np.random.shuffle(training_data)
        np.random.shuffle(testing_data)
        scaler.fit(training_data)
        TRAINING_INSTANCES = np.append(TRAINING_INSTANCES, scaler.transform(training_data)[None], axis=0)
        TESTING_INSTANCES = np.append(TESTING_INSTANCES, scaler.transform(testing_data)[None], axis=0)
    TRAINING_INSTANCES = TRAINING_INSTANCES.astype(np.float64)
    TESTING_INSTANCES = TESTING_INSTANCES.astype(np.float64)


def get_category(data):
    """Get the category of each column (categorical, numerical)"""
    category_set = np.asarray(['categorical'])
    for col in range(1, np.size(data[0])):
        if np.unique(np.unique(data[:, col])).size > 5:
            category_set = np.append(category_set, 'numerical')
        else:
            category_set = np.append(category_set, 'categorical')
    return category_set


def knn(category):
    global CLASSES
    CLASSES = np.unique(TRAINING_INSTANCES[CURRENT_K][:, 0])
    confusion_matrix = np.zeros((np.size(CLASSES), np.size(CLASSES)))
    for testing_instance in TESTING_INSTANCES[CURRENT_K]:
        distance = []
        for training_instance in TRAINING_INSTANCES[CURRENT_K]:
            distance_i = 0
            diff_attr = training_instance != testing_instance
            categorical = category == 'categorical'
            diff_category = diff_attr * categorical
            not_diff_category = diff_category != True
            distance_i += np.sum(
                np.power(training_instance[not_diff_category][1:] - testing_instance[not_diff_category][1:], 2))
            diff_category[0] = False
            distance_i += \
                np.size(training_instance[diff_category]) / np.size(category) * np.size(category[categorical])
            distance = np.append(distance, distance_i)
        idx = np.argpartition(distance, KNN)[:KNN]
        unique_value = np.unique(np.take(TRAINING_INSTANCES[CURRENT_K][:, 0], idx), return_counts=True)
        majority = unique_value[0][np.argmax(unique_value[1])]
        testing_class_idx = np.where(testing_instance[0] == CLASSES)[0][0]
        predicted_class_idx = np.where(majority == CLASSES)[0][0]
        confusion_matrix[testing_class_idx][predicted_class_idx] += 1
    # print(confusion_matrix)
    return confusion_matrix


def execution(category):
    global CURRENT_K
    global KNN
    acc = []
    f1 = []
    knn_set = [2 * i + 1 for i in range(25)]
    for knn_i in knn_set:
        KNN = knn_i
        accuracy = 0
        precision = 0
        recall = 0
        for k_i in range(K):
            CURRENT_K = k_i
            confusion_matrix = knn(category)
            accuracy += np.trace(confusion_matrix) / np.sum(confusion_matrix)
            precision_i = 0
            recall_i = 0
            class_size = np.size(CLASSES)
            for class_i in range(class_size):
                if np.sum(confusion_matrix[:, class_i]) == 0:
                    precision_i += 0
                else:
                    precision_i += confusion_matrix[class_i][class_i] / np.sum(confusion_matrix[:, class_i])
                recall_i += confusion_matrix[class_i][class_i] / np.sum(confusion_matrix[class_i])
            precision_i /= class_size
            recall_i /= class_size
            precision += precision_i
            recall += recall_i
        accuracy /= K
        precision /= K
        recall /= K
        acc.append(accuracy)
        f1.append(2 * precision * recall / (precision + recall))
        print("Accuracy: ", accuracy)
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("F1 Score: ", 2 * precision * recall / (precision + recall))
        print('\n')
    # plt.plot(knn_set, acc, label='Accuracy')
    # plt.plot(knn_set, f1, label='F1 Score')
    # plt.xlabel('K')
    # plt.ylabel('Percentage')
    # plt.legend()
    # plt.show()


def formalize(data, class_col):
    class_data = np.take(data, class_col, axis=1)
    boot_data = np.delete(data, class_col, axis=1)
    boot_data = np.insert(boot_data, 0, class_data, axis=1)
    return boot_data


def test_parkinsons():
    dataset = pd.DataFrame(pd.read_csv('datasets/parkinsons.csv')).values
    dataset = formalize(dataset, np.size(dataset[0]) - 1)
    category = get_category(dataset)
    stratify_data(dataset)
    execution(category)


def test_digits():
    digits = datasets.load_digits(return_X_y=True)
    digits_dataset_X = digits[0]
    digits_dataset_y = digits[1]
    dataset = np.insert(digits_dataset_X, 0, digits_dataset_y, axis=1)
    category = get_category(dataset)
    stratify_data(dataset)
    execution(category)


def test_titanic():
    le = pp.LabelEncoder()
    dataset = pd.DataFrame(pd.read_csv('datasets/titanic.csv')).values
    dataset = np.delete(dataset, 2, axis=1)
    category = get_category(dataset)
    for col_index in range(1, dataset[0].size):
        col = np.take(dataset, col_index, axis=1)
        unique_value = np.unique(col)
        if type(col[0]) == str or unique_value.size <= 5:
            col = le.fit_transform(col)
            dataset = np.delete(dataset, col_index, axis=1)
            dataset = np.insert(dataset, col_index, col, axis=1)
    stratify_data(dataset.astype(np.float64))
    execution(category)


def test_loan():
    le = pp.LabelEncoder()
    dataset = pd.DataFrame(pd.read_csv('datasets/loan.csv')).values
    dataset = np.delete(dataset, 0, axis=1)
    dataset = formalize(dataset, dataset[0].size - 1)
    category = get_category(dataset)
    for col_index in range(dataset[0].size):
        col = np.take(dataset, col_index, axis=1)
        unique_value = np.unique(col)
        if type(col[0]) == str or unique_value.size <= 5:
            col = le.fit_transform(col)
            dataset = np.delete(dataset, col_index, axis=1)
            dataset = np.insert(dataset, col_index, col, axis=1)
    stratify_data(dataset.astype(np.float64))
    execution(category)


if __name__ == "__main__":
    test_parkinsons()
    # test_loan()
    # test_titanic()
    # test_digits()
