import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

# Assuming 'data' is your feature matrix and 'labels' are corresponding labels

# Number of trials and folds
num_trials = 100
num_folds = 10

# Percentage of training data to use in the learning curve
training_percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Placeholder for accuracy values
accuracy_values = []

# Placeholder for mean and standard deviation values at each training percentage
mean_accuracies = []
std_dev_accuracies = []

# Loop over training percentages
for train_percentage in training_percentages:
    # Placeholder for trial accuracy values
    trial_accuracies = []

    # Loop over trials
    for trial in range(num_trials):
        # Shuffle data at the start of each trial
        data, labels = shuffle(data, labels)

        # Determine the number of samples to use based on the percentage
        num_samples = int(len(data) * train_percentage)

        # Split data into training and testing sets
        train_data = data[:num_samples]
        train_labels = labels[:num_samples]

        test_data = data[num_samples:]
        test_labels = labels[num_samples:]

        # Train the decision tree classifier
        classifier = DecisionTreeClassifier()
        classifier.fit(train_data, train_labels)

        # Make predictions on the test set
        predictions = classifier.predict(test_data)

        # Calculate accuracy and store in trial_accuracies
        accuracy = accuracy_score(test_labels, predictions)
        trial_accuracies.append(accuracy)

    # Calculate mean and standard deviation of trial_accuracies
    mean_accuracy = np.mean(trial_accuracies)
    std_dev_accuracy = np.std(trial_accuracies)

    # Store mean and std_dev in lists
    mean_accuracies.append(mean_accuracy)
    std_dev_accuracies.append(std_dev_accuracy)

# Plot the learning curve
plt.errorbar(training_percentages, mean_accuracies, yerr=std_dev_accuracies, fmt='o-', label='Test Accuracy')
plt.xlabel('Percentage of Training Data')
plt.ylabel('Accuracy')
plt.title('Learning Curve')
plt.legend()
plt.show()
