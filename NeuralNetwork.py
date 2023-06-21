import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import sklearn.preprocessing as pp

TRAINING_SIZE = 0
TRAINING_INSTANCES = []
TRAINING_CLASSES = []

TESTING_SIZE = 0
TESTING_INSTANCES = []
TESTING_CLASSES = []

K = 0
CURRENT_K = 0
ALPHA = 0
LAMBDA = 0
MAX_ITERATION = 1000
MIN_IMPROVEMENT = 0.0001


def evaluation(thetas):
    class_size = np.size(TESTING_CLASSES[CURRENT_K][0])
    confusion_matrix = np.zeros((class_size, class_size))
    for instance_i in range(TESTING_SIZE):  # For each testing instance
        output = forward_propagation(TESTING_INSTANCES[CURRENT_K][instance_i], thetas)[-1]  # Predicted output
        predicted_index = np.argmax(output)  # Predicted class
        true_index = np.argmax(TESTING_CLASSES[CURRENT_K][instance_i])  # True class
        confusion_matrix[true_index][predicted_index] += 1  # Row: True class   Col: Predicted class
    print("Confusion Matrix: \n", confusion_matrix)
    accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
    print("Accuracy: ", accuracy)
    precision = 0
    recall = 0
    for class_i in range(class_size):
        if np.sum(confusion_matrix[:, class_i]) == 0:
            precision += 0
        else:
            precision += confusion_matrix[class_i][class_i] / np.sum(confusion_matrix[:, class_i])
        recall += confusion_matrix[class_i][class_i] / np.sum(confusion_matrix[class_i])
    precision /= class_size
    recall /= class_size
    print("Precision: ", precision)
    print("Recall: ", recall)
    print('\n')
    return accuracy, precision, recall


def cost_function(thetas):
    J = 0  # Initialize cost
    for instance_i in range(TRAINING_SIZE):
        output = forward_propagation(TRAINING_INSTANCES[CURRENT_K][instance_i], thetas)[-1]  # Predicted output
        for i in range(len(output)):
            class_i = TRAINING_CLASSES[CURRENT_K][instance_i][i]
            J_i = -class_i * np.log(output[i]) - (1 - class_i) * np.log(1 - output[i])  # Add all elements of J_i
            J += J_i
    J /= TRAINING_SIZE  # Divided by the number of training instances
    S = 0
    for theta_i in thetas:
        S += np.sum(np.square(np.delete(theta_i, 0, axis=1)))  # Compute the term used to regularize the cost
    S *= LAMBDA / 2 / TRAINING_SIZE
    return J + S


def testing_cost_function(thetas):
    J = 0  # Initialize cost
    for instance_i in range(TESTING_SIZE):
        output = forward_propagation(TESTING_INSTANCES[CURRENT_K][instance_i], thetas)[-1]  # Predicted output
        for i in range(len(output)):
            class_i = TESTING_CLASSES[CURRENT_K][instance_i][i]
            J_i = -class_i * np.log(output[i]) - (1 - class_i) * np.log(1 - output[i])  # Add all elements of J_i
            J += J_i
    J /= TESTING_SIZE  # Divided by the number of training instances
    S = 0
    for theta_i in thetas:
        S += np.sum(np.square(np.delete(theta_i, 0, axis=1)))  # Compute the term used to regularize the cost
    S *= LAMBDA / 2 / TESTING_SIZE
    return J + S


def back_propagation(thetas):
    global ALPHA
    iteration = 0
    J = cost_function(thetas)
    J_improvement = J
    alpha = ALPHA
    # cost = [J]
    # training_num = [0]
    while np.abs(J_improvement) > MIN_IMPROVEMENT:
        if iteration < MAX_ITERATION:
            gradient = [np.zeros((np.size(thetas[i], axis=0), np.size(thetas[i], axis=1))) for i in
                        range(len(thetas))]
            for instance_i in range(TRAINING_SIZE):
                layers = forward_propagation(TRAINING_INSTANCES[CURRENT_K][instance_i], thetas)  # 1.1
                delta = [np.empty(0, np.float64) for i in range(len(layers) - 1)]  # 1.2
                delta_output = layers[-1] - TRAINING_CLASSES[CURRENT_K][instance_i]
                delta[-1] = np.append(delta[-1], delta_output)
                for layer_index in reversed(range(1, len(layers) - 1)):  # 1.3 for k = L - 1...2    Update delta
                    delta_i = np.dot(thetas[layer_index].T, delta[layer_index])[1:]
                    delta_i *= layers[layer_index][1:] * (1 - layers[layer_index][1:])
                    delta[layer_index - 1] = np.append(delta[layer_index - 1], delta_i)
                for layer_index in reversed(range(len(layers) - 1)):  # 1.4 for k = L - 1...1 Update gradient
                    gradient[layer_index] += np.array(
                        np.dot(delta[layer_index][None].T, layers[layer_index][None]), dtype=np.float64
                    )
            for gradient_index in range(len(gradient)):  # 2 Update gradients
                regularization = LAMBDA * thetas[gradient_index]
                regularization = np.insert(np.delete(regularization, 0, axis=1), 0, 0, axis=1)
                gradient[gradient_index] = (1 / TRAINING_SIZE) * (gradient[gradient_index] + regularization)
            for thetas_index in range(len(thetas)):  # 4. Update thetas
                thetas[thetas_index] -= alpha * gradient[thetas_index]
            new_J = cost_function(thetas)
            J_improvement = J - new_J
            J = new_J
            # cost.append(J)
            # training_num.append(training_num[-1] + TRAINING_SIZE)
            iteration += 1
            if J_improvement < 0:
                alpha *= 0.9
        else:
            break
    print("Total iteration: ", iteration)
    print("Final cost: ", J)
    # plt.plot(training_num, cost)
    # plt.xlabel('Number of training instances')
    # plt.ylabel('Cost J')
    # plt.show()
    return thetas


def plot_cost(x, y):
    plt.plot(x, y)
    plt.xlabel("Number of training instances")
    plt.ylabel("Cost J on testing instances")
    plt.show()


def forward_propagation(instance, thetas):
    layers = [np.insert(instance, 0, 1)]  # Initialize the first layer with bias neuron
    for theta_index in range(len(thetas)):
        # Compute the sigmoid of the input
        # print(-np.array(np.dot(thetas[theta_index], layers[theta_index].T), dtype=np.float64))
        sigmoid = 1 / (1 + np.exp(-np.array(np.dot(thetas[theta_index], layers[theta_index].T), dtype=np.float64)))
        # Add new layer to the layer list
        layers.append(np.insert(sigmoid, 0, 1) if theta_index != len(thetas) - 1 else sigmoid)
    return layers


def initialize_theta(layers_num):
    # Initialize the theta to random numbers sampled from Gaussian distribution
    theta = []
    for i in range(len(layers_num) - 1):
        theta_i = np.random.normal(0, 1, (layers_num[i + 1], layers_num[i] + 1))
        theta.append(theta_i)
    return theta


def onehot_encoder(classes):
    # Convert class into list   e.g. 1 -> [1, 0, 0]     3 -> [0, 0, 1]
    unique_classes = np.unique(classes)
    processed_y = np.empty((0, np.size(unique_classes)), np.ndarray)
    for class_i in classes:
        onehot_class = np.zeros(np.size(unique_classes))
        index = np.argwhere(unique_classes == class_i)[0]
        onehot_class[index] = 1
        processed_y = np.append(processed_y, [onehot_class], axis=0)
    return processed_y


def stratify_data(data):
    global TRAINING_SIZE
    global TESTING_SIZE
    global TRAINING_INSTANCES
    global TRAINING_CLASSES
    global TESTING_INSTANCES
    global TESTING_CLASSES

    class_value = np.unique(data[:, 0], return_counts=True)
    attr_size = np.size(data[0]) - 1
    class_size = np.size(class_value[0])

    stratified_data = np.empty((0, np.sum(np.array(class_value[1] / K, np.int64)), attr_size + 1), np.ndarray)
    split_class = []
    for class_i in np.nditer(class_value):
        sub_class_value = class_i[0]
        sub_class_condition = data[:, 0] == sub_class_value
        split_class.append(data[sub_class_condition])
    for k_i in range(K):
        sub_class_k = np.empty((0, attr_size + 1), np.ndarray)
        for class_index in range(len(split_class)):
            sub_class_size = np.size(split_class[class_index], axis=0)
            size_to_add = int(class_value[1][class_index] / K)
            row_num = np.random.choice(range(sub_class_size), size_to_add)
            sub_class_k = np.append(sub_class_k, np.take(split_class[class_index], row_num, axis=0), axis=0)
            split_class[class_index] = np.delete(split_class[class_index], row_num, axis=0)
        stratified_data = np.append(stratified_data, [sub_class_k], axis=0)

    TESTING_SIZE = np.size(stratified_data[0], axis=0)
    TRAINING_SIZE = TESTING_SIZE * (K - 1)
    TRAINING_INSTANCES = np.empty((0, TRAINING_SIZE, attr_size), np.ndarray)
    TRAINING_CLASSES = np.empty((0, TRAINING_SIZE, class_size), np.ndarray)
    TESTING_INSTANCES = np.empty((0, TESTING_SIZE, attr_size), np.ndarray)
    TESTING_CLASSES = np.empty((0, TESTING_SIZE, class_size), np.ndarray)
    for k_i in range(K):
        testing_data = stratified_data[k_i]
        training_data = np.reshape(np.delete(stratified_data, k_i, axis=0), (TRAINING_SIZE, attr_size + 1))
        scaler = pp.MinMaxScaler()
        scaler.fit(training_data[:, 1:])
        TRAINING_INSTANCES = np.append(TRAINING_INSTANCES, [scaler.transform(training_data[:, 1:])], axis=0)
        TRAINING_CLASSES = np.append(TRAINING_CLASSES, [onehot_encoder(training_data[:, 0])], axis=0)
        TESTING_INSTANCES = np.append(TESTING_INSTANCES, [scaler.transform(testing_data[:, 1:])], axis=0)
        TESTING_CLASSES = np.append(TESTING_CLASSES, [onehot_encoder(testing_data[:, 0])], axis=0)


def formalize(data, class_col):
    class_data = np.take(data, class_col, axis=1)
    boot_data = np.delete(data, class_col, axis=1)
    boot_data = np.insert(boot_data, 0, class_data, axis=1)
    return boot_data


def execution(hidden_layers):
    global CURRENT_K
    accuracy = []
    precision = []
    recall = []
    for k_i in range(K):
        CURRENT_K = k_i
        input_num = np.size(TRAINING_INSTANCES[k_i][0])
        output_num = np.size(TRAINING_CLASSES[k_i][0])
        theta = initialize_theta(np.append(np.insert(hidden_layers, 0, input_num), output_num))
        new_theta = back_propagation(theta)
        accuracy_k, precision_k, recall_k = evaluation(new_theta)
        accuracy.append(accuracy_k)
        precision.append(precision_k)
        recall.append(recall_k)
    average_accuracy = np.sum(accuracy) / K
    average_precision = np.sum(precision) / K
    average_recall = np.sum(recall) / K
    average_f1 = 2 * average_precision * average_recall / (average_precision + average_recall)
    print("Average Accuracy: ", average_accuracy)
    print("Average F1-Score: ", average_f1)


def test_parkinsons(hidden_layers):
    dataset = pd.DataFrame(pd.read_csv('datasets/parkinsons.csv')).values
    dataset = formalize(dataset, len(dataset[0]) - 1)
    stratify_data(dataset)
    execution(hidden_layers)


def test_digits(hidden_layers):
    digits = datasets.load_digits(return_X_y=True)
    digits_dataset_X = digits[0]
    digits_dataset_y = digits[1]
    dataset = np.insert(digits_dataset_X, 0, digits_dataset_y, axis=1)
    stratify_data(dataset)
    execution(hidden_layers)


def test_titanic(hidden_layers):
    le = pp.LabelEncoder()
    dataset = pd.DataFrame(pd.read_csv('datasets/titanic.csv')).values
    dataset = np.delete(dataset, 2, axis=1)
    for col_index in range(1, dataset[0].size):
        col = np.take(dataset, col_index, axis=1)
        unique_value = np.unique(col)
        if type(col[0]) == str or unique_value.size < 10:
            col = le.fit_transform(col)
            dataset = np.delete(dataset, col_index, axis=1)
            dataset = np.insert(dataset, col_index, col, axis=1)
    stratify_data(dataset.astype(np.float64))
    execution(hidden_layers)


def test_loan(hidden_layers):
    le = pp.LabelEncoder()
    dataset = pd.DataFrame(pd.read_csv('datasets/loan.csv')).values
    dataset = np.delete(dataset, 0, axis=1)
    dataset = formalize(dataset, dataset[0].size - 1)
    for col_index in range(dataset[0].size):
        col = np.take(dataset, col_index, axis=1)
        unique_value = np.unique(col)
        if type(col[0]) == str or unique_value.size < 5:
            col = le.fit_transform(col)
            dataset = np.delete(dataset, col_index, axis=1)
            dataset = np.insert(dataset, col_index, col, axis=1)
    stratify_data(dataset.astype(np.float64))
    execution(hidden_layers)


if __name__ == '__main__':
    K = 10
    ALPHA = 2
    LAMBDA = 0.25
    start = time.time()
    # test_parkinsons([12])
    # test_digits([20])
    # test_titanic([8, 4])
    # test_loan([8])
    print('Used Time:', time.time() - start)
