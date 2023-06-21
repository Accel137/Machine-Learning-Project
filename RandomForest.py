import random
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
from sklearn import datasets
import sklearn.preprocessing as pp


def get_attr_entropy(data, col, attr_set, category_set):
    # unique_value: [[0, 1, 2] [count1, count2, count3]]
    unique_value = np.unique(data[:, col], return_counts=True)
    entropy = 0
    if category_set[col] == 'categorical':
        for value in np.nditer(unique_value):
            sub_entropy = 0
            condition = data[:, col] == value[0]
            grouped_data = data[condition]
            sub_group = np.unique(grouped_data[:, 0], return_counts=True)
            for cls in np.nditer(sub_group):
                probability = cls[1] / value[1]
                sub_entropy -= probability * np.log2(probability)
            entropy += sub_entropy * value[1] / np.sum(unique_value[1])
    else:
        sub_entropy_1 = 0
        sub_entropy_2 = 0
        condition_1 = data[:, col] <= attr_set[col]
        condition_2 = data[:, col] > attr_set[col]
        grouped_data_1 = data[condition_1]
        grouped_data_2 = data[condition_2]
        sub_group_1 = np.unique(grouped_data_1[:, 0], return_counts=True)
        # print(sub_group_1)
        sub_group_2 = np.unique(grouped_data_2[:, 0], return_counts=True)
        if np.size(sub_group_1) != 0:
            for cls in np.nditer(sub_group_1):
                probability = cls[1] / np.size(grouped_data_1[:, 0])
                sub_entropy_1 -= probability * np.log2(probability)
            sub_entropy_1 *= np.size(grouped_data_1[:, 0]) / np.size(data[:, col])
        if np.size(sub_group_2) != 0:
            for cls in np.nditer(sub_group_2):
                probability = cls[1] / np.size(grouped_data_2[:, 0])
                sub_entropy_2 -= probability * np.log2(probability)
            sub_entropy_2 *= np.size(grouped_data_2[:, 0]) / np.size(data[:, col])
        entropy += sub_entropy_1 + sub_entropy_2
    return entropy


def get_attr_gini(data, col, attr_set, category_set):
    # unique_value: [[0, 1, 2] [count1, count2, count3]]
    unique_value = np.unique(data[:, col], return_counts=True)
    gini = 0
    if category_set[col] == 'categorical':
        for value in np.nditer(unique_value):
            sub_gini = 1
            condition = data[:, col] == value[0]
            grouped_data = data[condition]
            sub_group = np.unique(grouped_data[:, 0], return_counts=True)
            for cls in np.nditer(sub_group):
                probability = cls[1] / value[1]
                sub_gini -= np.square(probability)
            gini += sub_gini * value[1] / np.sum(unique_value[1])
    else:
        sub_gini_1 = 1
        sub_gini_2 = 1
        condition_1 = data[:, col] <= attr_set[col]
        condition_2 = data[:, col] > attr_set[col]
        grouped_data_1 = data[condition_1]
        grouped_data_2 = data[condition_2]
        sub_group_1 = np.unique(grouped_data_1[:, 0], return_counts=True)
        # print(sub_group_1)
        sub_group_2 = np.unique(grouped_data_2[:, 0], return_counts=True)
        if np.size(sub_group_1) != 0:
            for cls in np.nditer(sub_group_1):
                probability = cls[1] / np.size(grouped_data_1[:, 0])
                sub_gini_1 -= np.square(probability)
            sub_gini_1 *= np.size(grouped_data_1[:, 0]) / np.size(data[:, col])
        if np.size(sub_group_2) != 0:
            for cls in np.nditer(sub_group_2):
                probability = cls[1] / np.size(grouped_data_2[:, 0])
                sub_gini_2 -= np.square(probability)
            sub_gini_2 *= np.size(grouped_data_2[:, 0]) / np.size(data[:, col])
        gini += sub_gini_1 + sub_gini_2
    return gini


def get_class_entropy(data):
    unique_value = np.unique(data[:, 0], return_counts=True)
    entropy = 0
    for value in np.nditer(unique_value):
        entropy -= value[1] / data[:, 0].size * np.log2(value[1] / data[:, 0].size)
    return entropy


def select_attributes(num, data, attr_set, method):
    """
    Select the best attribute to be split
    :param num: The number of columns to be chosen
    :param data: Training set
    :param attr_set: Attribute set
    :param method: Information Gain or Gini
    :return: Attribute column 1, 2, ..., n
    """
    attr_col = np.arange(1)
    attr_col = np.append(attr_col, random.sample(range(1, data[0].size), num))
    # Save data with selected attributes
    attr_data = np.take(data, attr_col, axis=1)
    sub_attr_set = np.take(attr_set, attr_col)
    category_set = np.take(category, attr_col)
    if method == 'information_gain':
        entropy = np.zeros(attr_col.size - 1)
        for col in np.nditer(np.arange(1, num + 1)):
            entropy[col - 1] += get_attr_entropy(attr_data, col, sub_attr_set, category_set)
        return attr_col[np.argmin(entropy) + 1]
    elif method == 'gini':
        gini = np.zeros(attr_col.size - 1)
        for col in np.nditer(np.arange(1, num + 1)):
            gini[col - 1] += get_attr_gini(attr_data, col, sub_attr_set, category_set)
        return attr_col[np.argmin(gini) + 1]


def stratified(data):
    """
    Divide the data into k partitions
    :param data: Original data set
    :return: [[data partitions] * k]
    """
    class_value = np.unique(data[:, 0], return_counts=True)
    attr_size = np.size(data[0]) - 1
    stratified_data = np.empty((0, np.sum(np.array(class_value[1] / k, np.int64)), attr_size + 1), np.ndarray)
    split_class = []
    for class_i in np.nditer(class_value):
        sub_class_value = class_i[0]
        sub_class_condition = data[:, 0] == sub_class_value
        split_class.append(data[sub_class_condition])
    for k_i in range(k):
        sub_class_k = np.empty((0, attr_size + 1), np.ndarray)
        for class_index in range(len(split_class)):
            sub_class_size = np.size(split_class[class_index], axis=0)
            size_to_add = int(class_value[1][class_index] / k)
            row_num = np.random.choice(range(sub_class_size), size_to_add)
            sub_class_k = np.append(sub_class_k, np.take(split_class[class_index], row_num, axis=0), axis=0)
            split_class[class_index] = np.delete(split_class[class_index], row_num, axis=0)
        stratified_data = np.append(stratified_data, [sub_class_k], axis=0)
    stratified_data = np.array(stratified_data, dtype=np.float64)
    return stratified_data


def formalize(data, class_col):
    """Move class to the first column, then bootstrap the data
    """
    class_data = np.take(data, class_col, axis=1)
    boot_data = np.delete(data, class_col, axis=1)
    boot_data = np.insert(boot_data, 0, class_data, axis=1)
    return boot_data


def boot_strap(data, n, k_i):
    """
    Take kth stratified data as the testing set and generate n training set
    :param k_i:
    :param data: Original data set
    :param n: ntree parameter
    :return: [[training set] * n], [testing set]
    """
    testing_data = data[k_i]
    basic_training_data = np.delete(data, k_i, axis=0)
    width = np.size(basic_training_data, axis=0)
    length = np.size(basic_training_data, axis=1)
    depth = basic_training_data[0][0].size
    basic_training_data = np.reshape(basic_training_data, (width * length, depth))

    del_size = int(np.size(basic_training_data, axis=0) * 1 / 3)
    del_col = random.sample(range(0, width * length), del_size)
    basic_training_data = np.delete(basic_training_data, del_col, axis=0)
    basic_training_size = np.size(basic_training_data, axis=0)

    additional_index = random.sample(range(0, basic_training_size), del_size)
    additional_data = np.take(basic_training_data, additional_index, axis=0)
    training_data = np.append(basic_training_data, additional_data, axis=0)
    training_data_set = [training_data]
    for i in range(n - 1):
        additional_index = random.sample(range(0, basic_training_size), del_size)
        additional_data = np.take(basic_training_data, additional_index, axis=0)
        training_data = np.append(basic_training_data, additional_data, axis=0)
        training_data_set = np.append(training_data_set, [training_data], axis=0)
    return training_data_set, testing_data


def decision_tree(data, attr_set, method):
    best_attr = select_attributes(int(np.sqrt(data[0].size - 1)), data, attr_set, method)
    return decision_tree_recursion(data, best_attr, attr_set, 0, method)


def decision_tree_recursion(data, attr, attr_set, depth, method):
    """
    Build the tree
    :param data: data set to be build
    :param attr: Current best attribute
    :param attr_set:
    :param depth:
    :param method:
    :return: Decision tree: {attr_col: {attr_val1:{attr_col: {...}}, attr_val2:{}, ...}}
    """
    leaf_data = np.unique(data[:, 0], return_counts=True)

    if leaf_data[0].size == 1:  # When the data has only one class
        return {"leaf": leaf_data[0][0]}
    # When the criterion or the size of the data, or the depth of the tree reaches a boundary
    elif get_class_entropy(data) <= minimal_gain or np.sum(leaf_data[1]) <= minimal_size or depth >= maximal_depth:
        return {"leaf": leaf_data[0][np.argmax(leaf_data[1])]}
    node = {attr: {}}
    if category[attr] == 'categorical':
        unique_value = np.unique(data[:, attr], return_counts=True)
        for value in np.nditer(attr_set[attr]):
            condition = data[:, attr] == value
            sub_data = data[condition]
            if sub_data.size == 0:
                new_value = unique_value[0][np.argmax(unique_value[1])]
                new_class = np.unique(data[data[:, attr] == new_value][:, 0], return_counts=True)
                node[attr][int(value)] = {"leaf": new_class[0][np.argmax(new_class[1])]}
                continue
            best_attr = select_attributes(int(np.sqrt(sub_data[0].size - 1)), sub_data, attr_set, method)
            node[attr][int(value)] = decision_tree_recursion(
                sub_data, best_attr, attr_set, depth + 1, method)
    else:
        condition_1 = data[:, attr] <= attr_set[attr]
        condition_2 = data[:, attr] > attr_set[attr]
        sub_data_1 = data[condition_1]
        sub_data_2 = data[condition_2]
        if sub_data_1.size == 0:
            unique_value = np.unique(sub_data_2[:, 0], return_counts=True)
            new_class = unique_value[0][np.argmax(unique_value[1])]
            node[attr]['<' + str(attr_set[attr])] = {"leaf": new_class}
        else:
            new_attr_set_1 = attr_update(sub_data_1, attr_set)
            best_attr_1 = select_attributes(int(np.sqrt(sub_data_1[0].size - 1)), sub_data_1, new_attr_set_1, method)
            node[attr]['<' + str(attr_set[attr])] = decision_tree_recursion(
                sub_data_1, best_attr_1, new_attr_set_1, depth + 1, method)
        if sub_data_2.size == 0:
            unique_value = np.unique(sub_data_1[:, 0], return_counts=True)
            new_class = unique_value[0][np.argmax(unique_value[1])]
            node[attr]['>' + str(attr_set[attr])] = {"leaf": new_class}
        else:
            new_attr_set_2 = attr_update(sub_data_2, attr_set)
            best_attr_2 = select_attributes(int(np.sqrt(sub_data_2[0].size - 1)), sub_data_2, new_attr_set_2, method)
            node[attr]['>' + str(attr_set[attr])] = decision_tree_recursion(
                sub_data_2, best_attr_2, new_attr_set_2, depth + 1, method)
    return node


def decision_tree_classifier(tree, instance):
    """
    Classify an instance
    :param tree:
    :param instance:
    :return: The predicted value
    """
    attr = list(tree.keys())[0]
    sub_tree = tree[attr]
    while attr != "leaf":
        value = instance[attr]
        key = list(sub_tree.keys())[0]
        if type(key) != str:  # When the attribute is categorical
            attr = list(sub_tree[int(value)].keys())[0]
            sub_tree = sub_tree[int(value)][attr]
        else:  # When the attribute is numerical
            average = float(key[1:])
            if value <= average:
                attr = list(sub_tree[key].keys())[0]
                sub_tree = sub_tree[key][attr]
            else:
                attr = list(sub_tree['>' + key[1:]].keys())[0]
                sub_tree = sub_tree['>' + key[1:]][attr]
    return sub_tree


def n_tree(training_data_set, testing, attr_set, n, method):
    """
    Predict each instance in testing set and return the prediction
    :param training_data_set:
    :param testing:
    :param attr_set:
    :param n:
    :param method:
    :return: Array of the prediction [class1, class2, class3, ...]
    """
    tree_set = np.asarray([
        decision_tree(training_data_set[i], attr_set, method) for i in range(n)])
    prediction = []
    for instance in testing:
        individual_prediction = []
        for tree in tree_set:
            individual_prediction = np.append(
                individual_prediction, decision_tree_classifier(tree, instance))
        unique_value = np.unique(individual_prediction, return_counts=True)
        prediction = np.append(prediction, unique_value[0][np.argmax(unique_value[1])])
    return prediction


def k_fold(data, attr_set, n, method):
    """Perform k times predictions and evaluate the final performance"""
    stratified_data = stratified(data)  # Stratify the data with k
    unique_value = np.unique(data[:, 0])
    accuracy_set = []
    precision_set = []
    recall_set = []
    for i in range(k):
        training, testing = boot_strap(stratified_data, n, i)
        true_class = np.take(testing, 0, axis=1)
        prediction = [n_tree(training, testing, attr_set, n, method)]
        prediction = np.append(prediction, [true_class], axis=0)
        confusion_matrix = np.zeros([unique_value.size, unique_value.size], np.int64)
        for value in np.nditer(prediction, flags=['external_loop'], order='F'):
            confusion_matrix[int(value[1] - attr_set[0][0])][int(value[0] - attr_set[0][0])] += 1
        accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
        precision = 0
        recall = 0
        if unique_value.size > 2:
            for cls in unique_value:
                cls = int(cls - attr_set[0][0])
                precision += confusion_matrix[cls][cls] / np.sum(confusion_matrix[:, cls])
                recall += confusion_matrix[cls][cls] / np.sum(confusion_matrix[cls])
            precision /= unique_value.size
            recall /= unique_value.size
        else:
            precision = confusion_matrix[0][0] / np.sum(confusion_matrix[:, 0])
            recall = confusion_matrix[0][0] / np.sum(confusion_matrix[0, :])
        accuracy_set = np.append(accuracy_set, accuracy)
        precision_set = np.append(precision_set, precision)
        recall_set = np.append(recall_set, recall)
    average_accuracy = np.mean(accuracy_set)
    average_precision = np.mean(precision_set)
    average_recall = np.mean(recall_set)
    f1_score = 2 * average_precision * average_recall / (average_precision + average_recall)
    return average_accuracy, average_precision, average_recall, f1_score


def attr_processing(data):
    """
    Initiate the original attributes set
    :param data:
    :return: Array of attributes: [[0, 1], [0, 1, 2], x.xxx(average), xxxx(average)]
    """
    attrs_set = np.empty(data[0].size, np.ndarray)
    for col_i in range(data[0].size):
        unique_value = np.unique(data[:, col_i])
        if category[col_i] == 'categorical':
            attrs_set[col_i] = unique_value
        else:
            attrs_set[col_i] = np.mean(unique_value)
    return attrs_set


def attr_update(data, attr_set):
    """
    Update the average values in the attributes set when calculating with numerical column
    :param data:
    :param attr_set:
    :return: Array of updated attributes: [[0, 1], [0, 1, 2], x.xxx(average), xxxx(average)]
    """
    new_attrs_set = np.empty(data[0].size, np.ndarray)
    for col_i in range(data[0].size):
        unique_value = np.unique(data[:, col_i])
        if category[col_i] == 'numerical':
            new_attrs_set[col_i] = np.mean(unique_value)
        else:
            new_attrs_set[col_i] = attr_set[col_i]
    return new_attrs_set


def get_category(data):
    """Get the category of each column (categorical, numerical)"""
    category_set = np.asarray(['categorical'])
    for col in range(1, np.size(data[0])):
        if np.unique(np.unique(data[:, col])).size > 5:
            category_set = np.append(category_set, 'numerical')
        else:
            category_set = np.append(category_set, 'categorical')
    return category_set


def plot(x, y, label):
    plt.plot(x, y)
    plt.xlabel("ntree")
    plt.ylabel(label)
    plt.show()


def digits_data():
    digits = datasets.load_digits(return_X_y=True)
    digits_dataset_X = digits[0]
    digits_dataset_y = digits[1]
    data = np.insert(digits_dataset_X, 0, digits_dataset_y, axis=1)
    return data


def parkinsons():
    data = pd.DataFrame(pd.read_csv('datasets/parkinsons.csv').values.astype(np.float64)).to_numpy()
    data = formalize(data, np.size(data[0]) - 1)
    return data


def titanic():
    le = pp.LabelEncoder()
    data = pd.DataFrame(pd.read_csv('datasets/titanic.csv')).values
    data = np.delete(data, 2, axis=1)
    for col_index in range(1, data[0].size):
        col = np.take(data, col_index, axis=1)
        unique_value = np.unique(col)
        if type(col[0]) == str or unique_value.size < 10:
            col = le.fit_transform(col)
            data = np.delete(data, col_index, axis=1)
            data = np.insert(data, col_index, col, axis=1)
    data = np.array(data, dtype=np.float64)
    return data


def loan():
    le = pp.LabelEncoder()
    data = pd.DataFrame(pd.read_csv('datasets/loan.csv')).values
    data = np.delete(data, 0, axis=1)
    data = formalize(data, data[0].size - 1)
    for col_index in range(data[0].size):
        col = np.take(data, col_index, axis=1)
        unique_value = np.unique(col)
        if type(col[0]) == str or unique_value.size < 10:
            col = le.fit_transform(col)
            data = np.delete(data, col_index, axis=1)
            data = np.insert(data, col_index, col, axis=1)
    data = np.array(data, dtype=np.float64)
    return data


if __name__ == "__main__":
    k = 10
    minimal_gain = 0.01
    minimal_size = 5
    maximal_depth = 10

    start = time.time()

    '''Move the class to the first column'''
    # dataset = digits_data()
    dataset = parkinsons()
    # dataset = titanic()
    # dataset = loan()

    n_set = [1, 5, 10, 20, 30, 40, 50]
    category = get_category(dataset)
    attrs = attr_processing(dataset)

    final_accuracy = []
    final_precision = []
    final_recall = []
    final_f1_score = []
    for n_i in n_set:
        print(n_i)
        performance = k_fold(
            dataset, attrs, n_i, 'information_gain')
        final_accuracy.append(performance[0])
        final_precision.append(performance[1])
        final_recall.append(performance[2])
        final_f1_score.append(performance[3])
    print(time.time() - start)
    print("Final Accuracy:", final_accuracy)
    print("Final F1 Score:", final_f1_score)

    # plt.plot(n_set, final_accuracy, label='Accuracy')
    # plt.plot(n_set, final_f1_score, label='F1_Score')
    # plt.xlabel('ntree')
    # plt.ylabel('Percentage')
    # plt.legend()
    # plt.show()