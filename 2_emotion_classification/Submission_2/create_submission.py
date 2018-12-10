import pandas as pd
import numpy as np

y_pred = np.load('data/y_pred.npy')

df_output = pd.DataFrame(y_pred, columns = ['Category'])
df_output.insert(0, 'Id', range(0, df_output.size))
df_output.to_csv('ultimate_submission.csv', index = False)
print('Submission written')
