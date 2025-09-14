# SCT_DS_3
Bank Marketing Decision Tree Classifier
A machine learning project that predicts whether a customer will subscribe to a bank's term deposit based on their demographic and behavioral data using a Decision Tree Classifier.

 Project Overview
This project implements a Decision Tree Classifier to predict customer behavior in bank marketing campaigns. The model analyzes various customer attributes to determine the likelihood of a customer subscribing to a term deposit product.

Key Objectives:

Build a robust classification model using decision trees
Analyze customer demographics and behavioral patterns
Provide insights for targeted marketing strategies
Evaluate model performance using standard metrics
📊 Dataset
The project uses the Bank Marketing dataset from the UCI Machine Learning Repository. This dataset contains information about direct marketing campaigns (phone calls) of a Portuguese banking institution.

Dataset Characteristics:

Source: UCI Machine Learning Repository
Task: Binary Classification
Target Variable: y (whether the client subscribed to a term deposit)
Features: Customer demographics, campaign information, and economic indicators
Key Features Include:

Age, job, marital status, education
Default, housing loan, personal loan status
Contact communication type, duration
Campaign information (number of contacts, previous outcomes)
Economic indicators (employment variation rate, consumer confidence index)
Features
Decision Tree Implementation: Uses scikit-learn's DecisionTreeClassifier
Data Preprocessing: Handles categorical and numerical features
Model Evaluation: Comprehensive performance metrics including:
Classification Report (Precision, Recall, F1-Score)
Confusion Matrix
Accuracy Score
Reproducible Results: Fixed random state for consistent results
Error Handling: Robust error handling for file operations
🛠️ Installation
Clone the repository:
git clone https://github.com/yourusername/bank-marketing-decision-tree.git
cd bank-marketing-decision-tree
Create a virtual environment (recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install required packages:
pip install -r requirements.txt
Usage
Prepare your data:

Place your training dataset as training_dataset_03.csv
Place your testing dataset as testing_dataset_03_.csv
Update file paths in the script if necessary
Run the classifier:

python bank_marketing_classifier.py
Or use the Jupyter notebook:

jupyter notebook bank-marketing-decision-tree-classifier.ipynb
View results: The script will output:
Training progress
Classification report
Confusion matrix
📈 Model Performance
Based on the current implementation, the model achieves:

              precision    recall  f1-score   support

           0       0.90      0.84      0.87     11105
           1       0.09      0.15      0.11      1251

    accuracy                           0.77     12356
   macro avg       0.50      0.49      0.49     12356
weighted avg       0.82      0.77      0.79     12356
Key Metrics:

Overall Accuracy: 77%
Class 0 (No subscription): High precision (90%) and recall (84%)
Class 1 (Subscription): Low precision (9%) and recall (15%)
Performance Analysis: The model shows high accuracy for predicting customers who won't subscribe but struggles with the minority class (subscribers). This suggests class imbalance issues that could be addressed with techniques like:

SMOTE (Synthetic Minority Over-sampling)
Class weight balancing
Ensemble methods
Feature engineering
📁 Project Structure
bank-marketing-decision-tree/
│
├── bank-marketing-decision-tree-classifier.ipynb  # Main Jupyter notebook
├── bank_marketing_classifier.py                   # Python script version
├── training_dataset_03.csv                        # Training data
├── testing_dataset_03_.csv                        # Testing data
├── requirements.txt                                # Python dependencies
├── README.md                                       # Project documentation
├── .gitignore                                      # Git ignore file
└── results/                                        # Output results and visualizations
    ├── confusion_matrix.png
    ├── feature_importance.png
    └── classification_report.txt
📦 Dependencies
Python 3.7+
pandas
scikit-learn
numpy
matplotlib (for visualizations)
seaborn (for enhanced plotting)
jupyter (for notebook environment)
Install all dependencies using:

pip install -r requirements.txt
📊 Results
Current Model Performance
The model demonstrates strong performance for the majority class (non-subscribers)
Class imbalance significantly affects minority class prediction
Overall accuracy of 77% with room for improvement
Potential Improvements
Handle Class Imbalance: Implement SMOTE or adjust class weights
Feature Engineering: Create new features from existing data
Hyperparameter Tuning: Optimize tree depth, min_samples_leaf, etc.
Ensemble Methods: Try Random Forest or Gradient Boosting
Cross-Validation: Implement k-fold cross-validation for robust evaluation
 Contributing
Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

 Acknowledgments
UCI Machine Learning Repository for providing the Bank Marketing dataset
Scikit-learn community for the excellent machine learning tools
Contributors and maintainers of the open-source libraries used
Contact
If you have any questions, feel free to reach out:

Email: pallapudeepika85@gmail.com
LinkedIn:https://www.linkedin.com/in/pallapu-deepika/
GitHub: https://github.com/Deepika605
 If you found this project helpful, please give it a star! 
