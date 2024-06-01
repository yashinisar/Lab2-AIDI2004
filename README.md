# Breast Cancer Wisconsin (Diagnostic) Classification with Support Vector Machine

This project demonstrates the use of a Support Vector Machine (SVM) classifier to predict whether breast cancer is malignant or benign using the Breast Cancer Wisconsin (Diagnostic) dataset. The dataset is fetched using the `ucimlrepo` library.

## Dependencies

- pandas
- numpy
- scikit-learn
- ucimlrepo (for fetching the dataset)

## Dataset

The Breast Cancer Wisconsin (Diagnostic) dataset contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. These features describe the characteristics of the cell nuclei present in the image. The goal is to classify the breast cancer as either malignant or benign based on these features.

## Steps to Run the Code

1. **Import Libraries**: Essential libraries such as `pandas`, `numpy`, `train_test_split` from `sklearn`, `SVC`, and performance metrics are imported.

    ```python
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    ```

2. **Fetch and Load Data**: The dataset is fetched using the `ucimlrepo` library, and the features and targets are loaded into `X` and `y` respectively.

    ```python
    from ucimlrepo import fetch_ucirepo
    
    # Fetch the dataset
    breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)
    
    # Load data into pandas DataFrame
    X = breast_cancer_wisconsin_diagnostic.data.features
    y = breast_cancer_wisconsin_diagnostic.data.targets
    ```

3. **Split the Dataset**: The dataset is split into training and testing sets with 70% training data and 30% testing data to ensure that the model is trained and evaluated properly.

    ```python
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    ```

4. **Initialize the Support Vector Classifier**: A Support Vector Classifier is initialized with a linear kernel and a fixed random seed (`random_state=42`) for reproducibility.

    ```python
    svm_model = SVC(kernel='linear', random_state=42)
    ```

5. **Train the Model**: The model is trained on the training data.

    ```python
    svm_model.fit(X_train, y_train)
    ```

6. **Make Predictions**: Predictions are made on the test data.

    ```python
    y_pred = svm_model.predict(X_test)
    ```

7. **Evaluate the Model**: The model's performance is evaluated using accuracy, confusion matrix, and classification report.

    ```python
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print(f'Accuracy: {accuracy}')
    print('Confusion Matrix:')
    print(conf_matrix)
    print('Classification Report:')
    print(class_report)
    ```

## Results

The results of the model evaluation are printed, including:

- **Accuracy**: The proportion of correctly classified instances.
- **Confusion Matrix**: A table showing the true positives, true negatives, false positives, and false negatives.
- **Classification Report**: Detailed precision, recall, and F1-score for each class (malignant and benign).

## Conclusion

This project demonstrates how to build and evaluate a Support Vector Machine classifier for the Breast Cancer Wisconsin (Diagnostic) dataset. The model is capable of classifying breast cancer instances with high accuracy, providing valuable insights for medical diagnosis.

Feel free to explore and modify the code to experiment with different models and parameters!
