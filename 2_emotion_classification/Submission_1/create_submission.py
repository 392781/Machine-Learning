import pandas as pd
import numpy as np

# Load result predictions from directory
y_pred = np.load('data/y_pred.npy')

# Create a dataframe for the results with title "Category"
df_output = pd.DataFrame(y_pred, columns = ['Category'])
# Insert a new column before "Category" to write in "Id"
df_output.insert(0, 'Id', range(0, df_output.size))
# Save the CSV to directory with no written index
df_output.to_csv('ultimate_submission.csv', index = False)
print('Submission written')
