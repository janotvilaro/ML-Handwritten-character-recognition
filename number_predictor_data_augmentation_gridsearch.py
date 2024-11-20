import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from PIL import Image
import cv2
import os
from joblib import dump, load
from termcolor import colored

from termcolor import colored
import matplotlib.pyplot as plt
import os


# ********** NEW IMPORTS FOR DATA AUGMENTATION **********
import torch
from torchvision import transforms

# ********** FUNCTIONS **********

# Function to augment an image
def augment_image(image_array):
    image = Image.fromarray((image_array * 255).astype(np.uint8))
    augmented_image = augment_transform(image)
    return np.array(augmented_image)

# Function to preprocess the user-provided image
def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((28, 28), Image.LANCZOS)  # Resize to 28x28 pixels
    img = np.array(img)  # Convert image to array
    img = cv2.bitwise_not(img)  # Invert colors (0 black, 255 white)
    img = img / 255.0  # Normalize pixel values to [0, 1]

    # ********** APPLY DATA AUGMENTATION **********
    img = augment_image(img)

    img = img.reshape(1, -1)  # Turn the image into a vector
    return scaler.transform(img)  # Standardize based on training data
    
def predict_digit(image_path):
    img_input_preprocessed = preprocess_image(image_path)  # Preprocess the image input by the user
    prediction = svm.predict(img_input_preprocessed)  # Use SVM to predict the already preprocessed input image
    return prediction[0]
    
# Load the MNIST dataset
MNIST = datasets.fetch_openml('mnist_784', version=1)
X = MNIST.data / 255.0  # Divide each pixel by 255 since each pixel has 8 bits (2⁸)--> normalize image
y = MNIST.target.astype(np.int8)  # Target, that is the solutions of which number corresponds to each image

# Split the dataset into training, test and future prediction sets

# First split: separate out the 5% for future predictions
X_part, X_user, y_part, y_user = train_test_split(X, y, test_size=0.05, random_state=42)
long_future = len(y_user)
print(long_future)
# Second split: split the remaining 95% into 75% train and 20% test
X_train, X_test, y_train, y_test = train_test_split(X_part, y_part, test_size=0.21, random_state=42)

# Standardize the data
scaler = StandardScaler()  # Initialize the scaler
X_train_scaled = scaler.fit_transform(X_train)  # This results in a dataset where each feature has a mean of 0 and a standard deviation of 1.
X_test_scaled = scaler.transform(X_test)  # We apply the mean and standard deviation to the test data also, which have been stored internally
X_user_scaled = scaler.transform(X_user)

# ********** DATA AUGMENTATION TRANSFORMS **********
augment_transform = transforms.Compose([
    transforms.RandomRotation(20),  # Randomly rotate the image by up to 20 degrees
    transforms.RandomAffine(0, translate=(0.2, 0.2)),  # Randomly translate the image up to 20% of its width/height size
    transforms.ToTensor()  # Convert image to tensor
])

# Train and save the SVM model. If the model has not been trained yet, we do so, else we reload the NN data.
svm_model_path = 'svm_model_gridsearch.joblib'

param_grid = {'C':[0.1, 1, 10], 'kernel': ['linear','sigmoid']}
# Kernel and C are the parameters to be chosen from the Support Vector Classification algorithm
# Small C: A low value of C makes the decision surface smooth, meaning the model is less tolerant to misclassifications but more regularized (less likely to overfit).
# Large C: A high value of C aims to classify all training examples correctly, making the model more complex (more likely to overfit).

if not os.path.exists(svm_model_path):  # If the model has not been executed yet, we do so
    svm = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)  #refit an estimator with the found parametrers, verbose sets the info that will be printed.
    svm.fit(X_train_scaled, y_train)  # Fit model with the training data
    dump(svm, svm_model_path)  # Save the SVM model into the specified path, 'svm_model_path' using dump
    
    print(svm.best_params_)
else:
    svm = load(svm_model_path)

# Evaluate the model
y_predicted = svm.predict(X_test_scaled)  # Call SVM to predict the numbers associated to X_test. Sine refit= True, this prediction is already done with the best parameters found on the gridsearch.
print(classification_report(y_test, y_predicted))  # Check how good the model performed when applied to the test data (y_test vs y_pred)

# Predict the digit from the user-provided image
print(
colored('W', 'red'), colored('E', 'yellow'), colored('L', 'magenta'), colored('C', 'green'),
colored('O', 'blue'), colored('M', 'magenta'), colored('E', 'red'), colored('T', 'yellow'),
colored('O', 'green'), colored('T', 'blue'), colored('H', 'magenta'), colored('E', 'red'),
colored('N', 'yellow'), colored('U', 'green'), colored('M', 'blue'), colored('B', 'magenta'),
colored('E', 'red'), colored('R', 'yellow'), colored('P', 'green'), colored('R', 'blue'),
colored('E', 'magenta'), colored('D', 'red'), colored('I', 'yellow'), colored('C', 'green'),
colored('T', 'blue'), colored('O', 'magenta'), colored('R', 'red')
	)
	
# Define a menu and get user's input for the image path
directory_path = "/home/janot/handwriten_digits"
opt = "y"
while opt == "y" or opt == "Y":
    mnist_o_no = int(input("Would you like to predict your own image (type 0) or a digit from MNIST (type 1)? "))
    
    if mnist_o_no == 0:
        print("Available images: ", os.listdir(directory_path))  # List all that is in the specified directory
        image_filename = input("Please provide the image filename: ")
        image_path = os.path.join(directory_path, image_filename)
        
        if not os.path.isfile(image_path):
            print(colored('Please input a valid file name, consider adding .jpg', 'red'))
        else:
            digit = predict_digit(image_path)  # Call the function to predict digits, which will call the function to preprocess the input image
            print(colored('The predicted digit is:', 'green'), f" {digit}")
            
            # Display the image
            img = Image.open(image_path)
            plt.imshow(img, cmap='gray')
            plt.title(f"Predicted Digit: {digit}")
            plt.show()
    
    elif mnist_o_no == 1:
        index = int(input(f"You have available {long_future} images. Introduce a number between 0 and {long_future-1}: "))
        
        if 0 <= index < long_future:
            # Predict the digit for the chosen image
            y_user_predicted = svm.predict([X_user_scaled[index]])[0]
            y_user_real = y_user.iloc[index]  # Ensure using iloc for accessing pandas series correctly
            
            # Display the image
            plt.imshow(X_user.iloc[index].values.reshape(28, 28), cmap='gray')
            plt.title(f"Predicted Digit: {y_user_predicted}. Real Digit: {y_user_real}")
            plt.show()
        else:
            print("Please input a valid index.")
    
    else:
        print(colored('Please input a valid option', 'red'))
    
    opt = input("Would you like to predict another digit? (y/n): ")
    
    
