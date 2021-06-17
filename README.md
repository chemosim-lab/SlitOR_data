# SlitOR_data

Binary files and datasets for the SlitOR24-SlitOR25 paper:

> **Reverse chemical ecology in a moth: machine learning on odorant receptors identifies new behaviorally active agonists**  
> *Gabriela Caballero-Vidal [1], Cédric Bouysset [2], Jérémy Gévar [1], Hayat Mbouzid [1], Céline Nara [1], Julie Delaroche [1], Jérôme Golebiowski [2,3], Nicolas Montagné [1], Sébastien Fiorucci [2], & Emmanuelle Jacquin-Joly [1]*  
> [1] INRAE, Sorbonne Université, CNRS, IRD, UPEC, Université de Paris, Institute of Ecology and Environmental Sciences of Paris, Versailles, France  
> [2] Université Côte d’Azur, CNRS, Institut de Chimie de Nice UMR7272, Nice, France  
> [3] Department of Brain and Cognitive Sciences, Daegu Gyeongbuk Institute of Science and Technology, Daegu 711-873, South Korea

## Description

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

#### `sphere-exclusion.py`

Implementation of the sphere-exclusion algorithm. See the content of the file for the documentation.

#### **Applicability Domain**

For the similarity distance approach, the following constants (distance cutoff Dc and number of nearest-neighbors k) were used:
  - OR24: `Dc = 6.742`, `k=6`
  - OR25: `Dc = 9.422`, `k=9`
  
To obtain the reliability score for a molecule, given an array of descriptors, you can use the following code (implemented for Python v3.6.8 with numpy v1.16.1):
```python
import numpy as np

def reliability_score(v, training_set, k, Dc):
    '''Reliability score for a molecule given its descriptors.
    Based on Tropsha A, Gramatica P, & Gombar VK (2003) QSAR & Combinatorial Science 22:69-77
    
    Parameters
    ----------
    v : np.ndarray
        An array of descriptors of shape (N_descriptors, ) for a molecule
    training_set : np.ndarray
        An array of descriptors of shape (N_molecules, N_descriptors) for the full training set
    k : int
        Number of nearest neighbors used to calculate the average distance
    Dc : float
        Distance cutoff
    
    Notes
    -----
    The arrays of descriptors should be normalized using the supplied MinMaxScaler
    '''
    # euclidean distances with training set
    distances = np.linalg.norm(v - training_set, axis=1)
    # sort distances
    distances.sort()
    # Average distance between the molecule and its k-NN in the training set
    D = distances[:k].sum() / k
    # reliability score
    return 1 + (D - Dc) / Dc
