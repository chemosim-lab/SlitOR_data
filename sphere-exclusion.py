#!/usr/bin/env python
# coding: utf-8

"""Sphere exclusion module
=======================

:Author: Cédric Bouysset
:Year: 2021
:Copyright: MIT License
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance


class SphereExclusion:
    """Sphere exclusion algorithm

    Can be used for partitioning a dataset into a training and test set, or
    selecting a diverse subset, in a deterministic way.

    Attributes
    ----------
    X : pandas.DataFrame
        The features in the dataset after MinMax scaling between 0 and 1. This
        DataFrame is shortened as entries are pulled to the diverse subset or
        excluded by the MinMax procedure
    y : pandas.Series
        The endpoint values (activity, class...etc.). This Series is shortened
        identically to SphereExclusion.X
    df : pandas.DataFrame
        The original dataset
    columns : pandas.Index
        The columns used as features
    y_col : str
        Name of the column containing the endpoint
    selected : pandas.DataFrame
        The diverse subset selected by the algorithm. The original index labels
        are kept.
    verbose : bool
        Log some information to standard output if `True`

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset from which a diverse subset will be sampled
    y_col : str
        Name of the column containing the endpoint
    verbose : bool
        Log some information to standard output if `True`

    Example
    -------
    To get a diverse subset::

        >>> se = SphereExclusion(df, "Activity")
        >>> se.run(radius=4.0, init="center")
        >>> subset = se.selected.copy()

    To split the dataset in training and test sets::

        >>> se = SphereExclusion(df, "Activity")
        >>> train, test = se.split_train_test(radius=4.0, init="center")

    Notes
    -----
    This code isn't optimized for speed since the distance matrix is
    recalculated at every iteration.
    """
    def __init__(self, df, y_col, verbose=True):
        # get columns to normalize
        columns = df.columns.drop([y_col])
        # make a scaler trained on the complete dataset
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler = scaler.fit(df[columns])
        # normalize the dataset
        normalized = scaler.transform(df[columns])
        normalized = pd.DataFrame(normalized, columns=columns)
        self.X = normalized
        self.y = df[y_col].copy()
        self.df = df
        self.columns = columns
        self.y_col = y_col
        self.selected = pd.DataFrame(columns=columns)
        self.verbose = verbose

    def get_index(self, index):
        """Returns the normalized features of a particular row as a
        :class:`pandas.DataFrame`

        Parameters
        ----------
        index : int
            The corresponding row number. This might be different from the name
            displayed in the DataFrame
        """
        return self.X.iloc[index].to_frame().T
    
    def drop_index(self, index):
        """Removes a particular row from the pool of selectable instances

        Parameters
        ----------
        index : int
            The corresponding row number. This might be different from the name
            displayed in the DataFrame
        """
        name = self.X.iloc[index].name
        self.X.drop(index=name, inplace=True)
        self.y.drop(index=name, inplace=True)
    
    def select_index(self, index):
        """Assigns a particular row to the diverse subset and removes entries
        that are too similar from the pool of selectable instances

        Parameters
        ----------
        index : int
            The corresponding row number. This might be different from the name
            displayed in the DataFrame
        """
        x = self.get_index(index)
        self.selected = self.selected.append(x)
        self.drop_index(index=index)
        self.exclude(self.radius)

    def add_center(self):
        """Search for the entry that is the closest to the center of the
        normalized dataset and assigns it to the diverse subset. Also returns
        the row index of this entry.
        """
        center = pd.DataFrame([0.5]*len(self.columns), index=self.columns).T
        idx = distance.cdist(center.values, self.X.values).argmin()
        self.select_index(idx)
        return idx

    def exclude(self, radius):
        """Excludes entries that are too similar to the latest entry added to
        the diverse subset
        
        Similarity is based on the Euclidean distance calculated from the
        normalized features in the dataset. If the distance is below the given
        radius, the entry is marked as too similar and is excluded from the
        pool of entries.

        Parameters
        ----------
        radius : float
            Cutoff distance for the similarity
        """
        mat = distance.cdist(self.selected.tail(1).values,
                             self.X.values)
        idxs = np.flatnonzero(mat <= radius)
        for idx in sorted(idxs, reverse=True):
            self.drop_index(idx)

    def select_minmax(self):
        """Assigns the entry identified by the MinMax procedure to the diverse
        subset.

        MinMax will calculate the distance between the entries that
        are already in the diverse subset, and the ones in the pool. It will
        then only keep the maximum distance for each pool entry, and select the
        minimum among these max distances
        """
        mat = distance.cdist(self.selected.values, self.X.values)
        idx = mat.max(axis=0).argmin()
        self.select_index(idx)
    
    def run(self, radius, init="center", n_max=.25):
        """Run the sphere-exclusion algorithm on the dataset to prepare a
        diverse subset.

        Parameters
        ----------
        radius : float
            Radius of the exclusion sphere in the normalized feature space
        init : str or int
            If `init="center"`, the entry closest to the center of the dataset
            is used to initiate the algorithm. Else, the corresponding row
            index is used (might not correspond the row label)
        n_max : int or float
            If n_max is of type `int`, corresponds to the maximum number of
            entries that will be in the diverse subset. If `float`, this is
            assumed to be the desired size of the diverse subset as a fraction
            of the original dataset.

        Notes
        -----
        If the `radius` is too large, the size of the resulting diverse subset
        might be smaller than the desired `n_max` as too many entries are
        excluded.
        """
        self.radius = radius
        if isinstance(n_max, float):
            n_max = int(n_max * len(self.X))
            if self.verbose:
                print(f"Maximum number of entries set to {n_max}")
        if init == "center":
            init = self.add_center()
            if self.verbose:
                print(f"Initiated with entry #{init} (center)")
        else:
            if self.verbose:
                print(f"Initiating with entry #{init}")
            self.select_index(init)
        n = len(self.selected)
        i = 0
        while (n < n_max) and (len(self.X) > 0):
            i += 1
            self.select_minmax()
            n = len(self.selected)
            if self.verbose:
                print(f"\rIteration {i:3d} - selected {n:3}, {len(self.X):4d} remaining", end="")
    
    def split_train_test(self, *args, **kwargs):
        """Splits the original dataset into a training set and a diverse test
        set. For the list of parameters, see :meth:`SphereExclusion.run`

        Returns
        -------
        train, test : pandas.DataFrame, pandas.DataFrame
            The resulting datasets, not normalized, and with their original
            index labels kept
        """
        self.run(*args, **kwargs)
        mask = [i for i in self.df.index if i not in self.selected.index]
        train = self.df.iloc[mask].copy()
        test = self.df.iloc[self.selected.index].copy()
        return train, test
