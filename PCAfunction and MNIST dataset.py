import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

# Function implementing PCA:

# step1: first centre the data by subtracting the mean
# step2: calculate the covariance matrix
# step3: compute the eigenvalues and eigenvectors
# step4: sort Eigenvalues in descending order
# step5: select a subset from the rearranged Eigenvalue matrix
# step6: transform the data

def PCA(X, num_components):
    # Step1: centre the data by subtracting the mean
    # axis = 0 means along the column and axis = 1 means working along the row.
    X_centred = X - np.mean(X, axis=0)

    # Step2: calculate the covariance matrix
    # rowvar is True defautly, if rowvar is true, then every row represents a variable
    Covariance_matrix = np.cov(X_centred, rowvar = False)

    #Step3: compute the eigenvalues and eigenvectors
    # ducumentation: https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html
    eigenvalues, eigenvectors = np.linalg.eig(Covariance_matrix)

    # step4: sort Eigenvalues in descending order
    sorted_index_ascending = np.argsort(eigenvalues)
    sorted_index_descending = sorted_index_ascending[::-1]
    sorted_eigenvalues = eigenvalues[sorted_index_descending]
    sorted_eigenvectors = eigenvectors[:, sorted_index_descending]

    # step5: select a subset from the rearranged Eigenvalue matrix
    eigenvalues_chosen = eigenvalues[sorted_index_descending[0:num_components]]
    eigenvectors_chosen = eigenvectors[:, sorted_index_descending[0:num_components]]

    # step6: transform the data
    X_reduced = np.dot(X_centred, eigenvectors_chosen)
    return X_reduced, eigenvalues_chosen, eigenvalues


# Run PCA function on the MNIST dataset:

# prepare the data
data = pd.read_csv('MNIST_train.csv', header=0)
# data is from column 1 to the end
X = data.iloc[:, 1:]
# labels are in the first column
label = data.iloc[:, 0]

# Applying PCA function to X (2 principal components)
X_reduced, eigenvalues_chosen, eigenvalues = PCA(X, 2)

# Create a pandas dataframe with column names
principal_df = pd.DataFrame(X_reduced, columns = ['PC1','PC2'])

# Concatenate labels with principal_df, horizontally
final_df = pd.concat([principal_df, pd.DataFrame(label)], axis=1)

# Print to check if the format is right
print(final_df)

# percentage of the data variance is accounted for by the first two principal components
print("The percentage of the data variance accounted for by the first two principal components is: ",
      round((eigenvalues_chosen.sum() / eigenvalues.sum()),4))

# Plot the results
plt.figure()
sb.scatterplot(data = final_df, x = 'PC1', y = 'PC2', hue = 'labels', s = 10, palette= 'deep')
plt.show()