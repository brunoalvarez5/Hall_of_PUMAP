import numpy as np
import pandas as pd

#lets now safe X in a file so we can test the other executions with the same vectors 

X = np.random.rand(12, 32)
print(X)
df = pd.DataFrame(X)
df.to_csv("data_to_replicate_tests.csv", index = False)