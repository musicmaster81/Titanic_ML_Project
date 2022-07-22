# Perhaps one of the most tragic events in the past few-hundred years was the sinking of the Titanic cruise ship.
# Originally thought to be indestructible, we are all well aware that it sank on its maiden voyage. However, thanks to
# a dataset compiled by the people at Kaggle, we have historical data on the background of the victims of this tragic
# event. The goal of this project is to use that dataset to create a classification machine learning algorithm that can
# predict ,with some level of certainty, who would have survived the wreck based upon their "features."

# First, we import the necessary libraries to construct our algorithm
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures

# A fundamental concept in Machine Learning is to split your dataset into two groups: a training set and a test set. The
# training set is used to generate a model, while our test set is used to apply that model to determine its accuracy. As
# a result, we define the file paths to each set on our computer.
training_set_path = r'C:\Python\Data Sets\train.csv'
test_set_path = r'C:\Python\Data Sets\test.csv'

# We then begin with our training dataset. We start by reading our training file into a dataframe.
training_set = pd.read_csv(training_set_path, index_col='PassengerId')

# Now, let's take a look at our dataframe. We need to have an idea of what it looks like in order to be able to choose
# an appropriate set of features.
pd.set_option('display.max_columns', 15)  # This allows us to view all the column within our PyCharm IDE
pd.set_option('display.width', 320)  # Formats our rows correctly.
print(training_set.head(15))
print("\n")

# Right off the bat, we notice that our data set already has a "survived" column. Regardless of the features that we
# choose, we will use this column as our target vector.

# The next part of our analysis will be to perform some data cleaning. From the display of our dataframe, we can see
# there are a good deal of null values. Let's see how many Null values there for each column.
print(training_set.isnull().sum())
print("\n")

# There are a lot of null values in the 'Age' column of our dataframe. Because age will most likely be a feature that we
# choose to incorporate into our algorithm, we need to deal with this. Let's replace the NaN values with the mean of the
# Age column. We also replace the answers in the Fare column with its mean as well.
training_set['Age'].replace(np.nan, training_set['Age'].mean(), inplace=True)
training_set['Fare'].replace(np.nan, training_set['Fare'].mean(), inplace=True)
print(training_set.isnull().sum())  # This helps us check to make sure our cleaning step above worked.
print("\n")

# Recall that our matrix contains columns that have strings for their values. Perhaps the most useful column to quantify
# is the Sex column. Let's assign the value of females to "0" and males to "1".
training_set['Sex'].replace(['female', 'male'], [0, 1], inplace=True)
print(training_set)

# Now that we have a fairly clean dataset, we progress to choosing our set of features to train our model. To do so, we
# create a correlation heatmap to see which columns are correlated strongly with not just one another, but with the
# survived column.
correlation_matrix = training_set.corr()  # Creates a correlation dataframe
high_corr_features = correlation_matrix.index  # We extract the index from the dataframe created above
plt.figure(figsize=(20, 20))  # Determine our chart size
heat_map = sns.heatmap(training_set[high_corr_features].corr(), annot=True, cmap='RdYlGn')  # Use seaborn to create map
plt.title("Correlation Heatmap of Features")  # Give our graph a title
plt.show()  # Display our heatmap

# Now that we have decided upon our features, we are now ready to create our feature matrix and our target vector.
train_y = training_set['Survived']
train_X = training_set[['Fare', 'Pclass', 'Sex']]
print(train_y.head())
print("\n")
print(train_X.head(10))

# One concept that we wish to apply to our model is feature normalization. While our dataset is relatively small, I want
# to show off that I understand what this concept is and how to apply it. Feature normalization helps our model by
# ideally scaling all of our features to a value between 0 and 1. This is accomplished by subtracting the mean of each
# column of our examples and dividing by the column's standard deviation. The following function does just that:


def normalize(matrix):
    for column in matrix.columns:  # We iterate through each column in the matrix passed as an argument
        mean = matrix[column].mean()  # We calculate the mean of each column
        st_dv = matrix[column].std()  # we calculate the standard deviation of each column
        matrix[column] = matrix[column] - mean  # We subtract the mean from each column
        matrix[column] = matrix[column]/st_dv  # We divide each column by the standard deviation
    return matrix  # We then return our normalized matrix


# We now apply our function created above to our training matrix:
norm_X = normalize(train_X)

# One thing that may be useful to try is creating Polynomial features. This is done by taking each feature and computing
# the product of itself, and the other features up to the n-th degree. This can be done with the Polynomial Features
# class form sklearn:
poly = PolynomialFeatures(degree=2)  # We start off with a polynomial of degree 2
poly_X = poly.fit_transform(norm_X)

# We now proceed to creating our model. Since this is a classic classification problem, the first model that we will use
# will be Logistic Regression. This is because our desired output is binary. The passengers either died or survived. In
# the description of the dataset, a value of "1" means an individual survived whereas a value of "0" means they died.
# We start off by defining the Logistic Regression class from scikit-learn and train our model on our feature matrix.
log_reg = LogisticRegression()
log_reg.fit(poly_X, train_y)

# The logistic regression class creates our model by selecting the appropriate weights (often referred to as Theta/Beta
# values) that generate an appropriate "decision boundary". This boundary is used as the cut-off for our model to
# determine which passengers survived and which passengers died.

# We know create our test feature matrix and prediction target vector using the test dataset provided to us by Kaggle.
test_set = pd.read_csv(test_set_path, index_col='PassengerId')  # Creates a dataframe from our CSV file

# Similarly, as with our training set, let's check for Null values and fill those values in with the mean.
print(test_set.isnull().sum())
print("\n")
test_set['Fare'].replace(np.nan, test_set['Fare'].mean(), inplace=True)
test_set['Age'].replace(np.nan, test_set['Age'].mean(), inplace=True)

# Next, we choose our feature class.
test_X = test_set[['Fare', 'Pclass', 'Sex']]  # Creates our feature matrix using 3 columns.

# We use the same procedure as before to change the male and female strings to binary 1's and 0's.
test_X['Sex'].replace(['female', 'male'], [0, 1], inplace=True)

# We check to see that our changes worked:
print(test_X.isnull().sum())
print("\n")

# We then normalize our matrix
test_norm_X = normalize(test_X)

# We then create a polynomial feature matrix for our test data
poly_test = poly.fit_transform(test_norm_X)

# We now apply our model to the test set to generate our predictions.
predicted_y = log_reg.predict(poly_test)

# Let's take a look to make sure our vector created above only has values of 0 and 1
print(predicted_y)
print("\n")

# Success! We know place our predictions back into a series with the corresponding passenger ID of our prediction:
prediction_series = pd.Series(data=predicted_y, index=test_X.index)
print(prediction_series)

# We now create a csv file that contains our answers and submit it to Kaggle for grading. Results are given below
prediction_series.to_csv(path_or_buf=r'C:\Python\Data Sets\titanic_predictions.csv', header=['Survived'])

# =================================================== CONCLUSION ======================================================#
# According to Kaggle, this model is 77% accurate, which isn't bad, but it isn't particularly great either. I've linked
# to the competition submission site in the portfolio Titanic project section. Feel free to look up "musicmaster81" in
# the submission section to validate my score. Note that I will consistently be making improvement to this model, but
# have posted it to my portfolio to showcase that I understand the fundamental concepts of Logistic Regression. Feel
# free to download my answer CSV file and submit it to Kaggle for verification.

