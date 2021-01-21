# SlitOR_data

Binary files and datasets for the SlitOR24-SlitOR25 paper.

#### `supp data 2 SlitOR24+25_dataset.xlsx`

List of training and test set compounds with their associated name, CAS, SMILES, and activity.

#### `descriptors_list_*.csv`

CSV file containing the list of Dragon (v6.0.38) descriptors used by each model.

#### `minmaxscaler_*.joblib`

Saved MinMax scaler from the scikit-learn (v0.20.2) Python package (`sklearn.preprocessing.MinMaxScaler`).  
To load the scaler, use the joblib Python package (`joblib.load`).

#### `model_*.weka`

Saved model from Weka (v3.8.2):
  - `weka.classifiers.lazy.IBk` for OR25
  - `weka.classifiers.trees.RandomForest` for OR24.

#### **Applicability Domain**

For the similarity distance approach, the following constants (distance cutoff Dc and number of nearest-neighbors k) were used:
  - OR24: `Dc = 6.742`, `k=6`
  - OR25: `Dc = 9.422`, `k=9`
  
  To obtain the reliability score for a molecule, given an array of descriptors, you can use the following code:
```python
import numpy as np

def reliability_score(v, training_set, k, Dc):
    '''Reliability score for a compound given its descriptors
    
    Parameters
    ----------
    v : np.ndarray
        An array of descriptors of shape (N_descriptors, ) for a molecule
    training_set : np.ndarray
        An array of descriptors of shape (N_molecules, N_descriptors) for the full training set
    
    Notes
    -----
    The array of descriptors should be normalized using the supplied MinMaxScaler
    '''
    # euclidean distances with training set
    distances = np.linalg.norm(v - training_set, axis=1)
    # sort distances
    distances.sort()
    # Average distance between compound i and its k-NN in train: avg_Di
    D = distances[:k].sum() / k
    return 1 + (D - Dc) / Dc
