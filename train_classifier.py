import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load data from pickle file
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Extract data and labels
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Check label distribution
unique, counts = np.unique(labels, return_counts=True)
class_counts = dict(zip(unique, counts))
print(f"Class distribution: {class_counts}")

# Filter out classes with fewer than 2 samples
valid_classes = [key for key, value in class_counts.items() if value >= 2]
valid_indices = np.isin(labels, valid_classes)

# Update data and labels by removing invalid classes
data = data[valid_indices]
labels = labels[valid_indices]

# Check if there are enough samples after filtering
if len(data) == 0:
    print("Error: Not enough data remaining after filtering!")
else:
    # Split data into training and testing sets without stratification
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True)

    # Initialize the model
    model = RandomForestClassifier()

    # Train the model
    model.fit(x_train, y_train)

    # Make predictions
    y_predict = model.predict(x_test)

    # Calculate accuracy
    score = accuracy_score(y_test, y_predict)

    # Print the accuracy
    print(f'{score * 100:.2f}% of samples were classified correctly!')

    # Save the trained model
    with open('model.p', 'wb') as f:
        pickle.dump(model, f)
