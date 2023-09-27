---
---
# Bagging
Bagging, short for Bootstrap Aggregating, is an ensemble learning technique designed to improve the stability and accuracy of machine learning models.
It accomplishes this by training multiple instances of a base learning algorithm on different subsets of the training data and then combining their predictions.

NOTE: (in bagging only one algorithm will be used for sample training)
### The key advantages of Bagging include:

**Reduced Overfitting:**
- Bagging helps reduce overfitting, especially for complex models like decision trees, by training on different subsets of the data.

**Improved Stability:**
- The ensemble of models tends to be more stable and less sensitive to small variations in the training data.
  
**Handling of Outliers and Noisy Data:**
- Bagging can handle noisy data and outliers effectively because it reduces the impact of individual data points.

**Parallelization:**
- The base models can be trained in parallel, which makes Bagging computationally efficient.
  
**Versatility:**
- Bagging can be applied to a wide range of base learning algorithms, making it a versatile ensemble method.


Bagging is particularly useful when the base learner is prone to overfitting or when we want to increase the stability and generalization of the model.
It's widely used in practice to improve the performance of machine learning models. One of the most well-known algorithms that use Bagging is the Random Forest.

**Here's how the Bagging process typically works:**

![bagging](https://github.com/MANOJ-S-NEGI/decisiontree_randomforest_classification/assets/99602627/6ef4aed8-6080-4ad8-b0d1-4b7d62327396)

**1. Bootstrap Sampling:**
  - Randomly select multiple subsets (with replacement) from the training data. Each subset is called a "bootstrap sample." These samples are used to train individual base models.

**2. Base Learner Training:**
  - Train a base learning algorithm (e.g., Random Forest, Decision tree, neural network, etc.) on each bootstrap sample. This results in multiple different models.
  

**3. Aggregation (Combination):**
 - Combine the predictions from these base models. The aggregation method depends on the type of problem (classification, regression, etc.).
 - For classification problems, the most common aggregation method is to take a majority vote (mode) among the predictions of the base models.
 - For regression problems, the predictions are usually averaged.

**4. Final Prediction:**
 - The final prediction is obtained by aggregating the predictions of all the base models.

```
## param for random forest
parameters_RF = {
            'criterion': 'entropy',
            'max_depth': 10,
            'max_features': 'log2',
            'min_samples_leaf': 2,
            'min_samples_split': 10,
            'n_estimators': 100
            }
for estimate in range(1,30,1):    
    # Define the Bagging Classifier with multiple base learners
    bagging_model = BaggingClassifier(RandomForestClassifier(**parameters_RF), n_estimators=estimate)

    # Train the Bagging model
    bagging_model.fit(x_train_res, y_train_res)

    # Predict on the test set
    y_pred = bagging_model.predict(x_test)


```


   ---
   ---
   
 # votingclassifier

 A Voting Classifier is an ensemble learning method that combines the predictions of multiple base estimators (machine learning models) and predicts the class label by taking a vote.
 It's applicable to both classification and regression problems.

 **There are two main types of voting in a Voting Classifier:**

**1. Hard Voting:**
- In hard voting, each base estimator (model) predicts the class label, and the final prediction is determined by a majority vote. The class label with the most votes becomes the final prediction.
- In the context of classification, the class that receives the most votes is chosen as the final predicted class label.

2. Soft Voting:
- In soft voting, each base estimator predicts the probability distribution over all the classes. The final prediction is determined by averaging these probabilities and then selecting the class with the highest average probability.
- The final prediction is then determined by averaging these probabilities across all the base estimators and selecting the class with the highest average probability.
![votting](https://github.com/MANOJ-S-NEGI/decisiontree_randomforest_classification/assets/99602627/fd624581-13b1-451c-bcd9-898b3c6dea81)



```
## importing algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

## param for random forest
parameters_RF = {
            'criterion': 'entropy',
            'max_depth': 10,
            'max_features': 'log2',
            'min_samples_leaf': 2,
            'min_samples_split': 10,
            'n_estimators': 100
            }

parameters_DT = {
            'criterion': 'entropy',
            'max_depth': 12,
            'max_features': 'log2',
            'min_samples_leaf': 14,
            'min_samples_split': 8,
            'splitter': 'best'
            }
            
# Define multiple base learners (Decision Tree, Random Forest)
base_learners = [
                ('Decision Tree', DecisionTreeClassifier(**parameters_DT)),
                ('Random Forest', RandomForestClassifier(**parameters_RF)),
                
            ]
# Define the ensemble model using VotingClassifier
ensemble_model = VotingClassifier(estimators = base_learners, voting='hard')
## training model:
vote_model = ensemble_model.fit(x_train_res, y_train_res)

## prediction:
y_pred = vote_model.predict(x_test)
  ```
