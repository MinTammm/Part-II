import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

# Assuming 'data' is your feature matrix and 'labels' are corresponding labels

# Number of trials and folds
num_trials = 100
num_folds = 10

# Placeholder for accuracy values
accuracy_values = []

# Loop over trials
for trial in range(num_trials):
    # Shuffle data at the start of each trial
    data, labels = shuffle(data, labels)

    # Split data into folds
    folds_data = np.array_split(data, num_folds)
    folds_labels = np.array_split(labels, num_folds)

    # Placeholder for trial accuracy values
    trial_accuracies = []

    # Loop over folds
    for i in range(num_folds):
        # Prepare training and testing sets
        test_data = folds_data[i]
        test_labels = folds_labels[i]

        train_data = np.concatenate(folds_data[:i] + folds_data[i + 1:])
        train_labels = np.concatenate(folds_labels[:i] + folds_labels[i + 1:])

        # Train the decision tree classifier
        classifier = DecisionTreeClassifier()
        classifier.fit(train_data, train_labels)

        # Make predictions on the test set
        predictions = classifier.predict(test_data)

        # Calculate accuracy and store in trial_accuracies
        accuracy = accuracy_score(test_labels, predictions)
        trial_accuracies.append(accuracy)

    # Store trial_accuracies in accuracy_values
    accuracy_values.extend(trial_accuracies)

# Calculate mean and standard deviation of accuracy values
mean_accuracy = np.mean(accuracy_values)
std_dev_accuracy = np.std(accuracy_values)

# Print the results
print(f"Mean Accuracy: {mean_accuracy}")
print(f"Standard Deviation of Accuracy: {std_dev_accuracy}")
