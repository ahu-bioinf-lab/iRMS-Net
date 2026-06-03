import csv
from itertools import islice
from pathlib import Path
import warnings

import networkx as nx
import numpy as np
import pandas as pd
from rdkit import Chem

from utils_test import *

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'


def get_cell_feature(cellId, cell_features):
    for row in islice(cell_features, 0, None):
        if row[0] == cellId:
            return row[1:]


def atom_features(atom):
    return np.array(
        one_of_k_encoding_unk(
            atom.GetSymbol(),
            ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
             'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
             'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
             'Pt', 'Hg', 'Pb', 'Unknown']
        )
        + one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        + one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        + one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        + [atom.GetIsAromatic()]
    )


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception(f"input {x} not in allowable set {allowable_set}")
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smile}")

    c_size = mol.GetNumAtoms()
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])

    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return c_size, features, edge_index


def creat_data(datasets, cellfile):
    (DATA_DIR / 'processed').mkdir(parents=True, exist_ok=True)

    cell_features = []
    with open(cellfile) as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            cell_features.append(row)
    cell_features = np.array(cell_features)
    print('strain_features', cell_features.shape)

    compound_iso_smiles = []
    df = pd.read_csv(DATA_DIR / 'smiles' / 'smile.csv')
    compound_iso_smiles += list(df['smile'])
    compound_iso_smiles = set(compound_iso_smiles)

    smile_graph = {}
    print('compound_iso_smiles', len(compound_iso_smiles))
    for smile in compound_iso_smiles:
        smile_graph[smile] = smile_to_graph(smile)

    processed_drug1_file = DATA_DIR / 'processed' / f'{datasets}_drug1.pt'
    processed_drug2_file = DATA_DIR / 'processed' / f'{datasets}_drug2.pt'

    if (not processed_drug1_file.is_file()) or (not processed_drug2_file.is_file()):
        df = pd.read_csv(DATA_DIR / f'{datasets}.csv')
        drug1 = np.asarray(list(df['DrugA']))
        drug2 = np.asarray(list(df['DrugB']))
        cell = np.asarray(list(df['strain']))
        label = np.asarray(list(df['interaction']))

        print('start creating data')
        print(datasets)
        TestbedDataset(
            root=str(DATA_DIR),
            dataset=datasets + '_drug1',
            xd=drug1,
            xt=cell,
            xt_featrue=cell_features,
            y=label,
            smile_graph=smile_graph
        )
        TestbedDataset(
            root=str(DATA_DIR),
            dataset=datasets + '_drug2',
            xd=drug2,
            xt=cell,
            xt_featrue=cell_features,
            y=label,
            smile_graph=smile_graph
        )
        print('data created successfully')
        print('preparing', datasets, 'in pytorch format')
    else:
        print(f'{processed_drug1_file} and {processed_drug2_file} already exist')


if __name__ == "__main__":
    cellfile = DATA_DIR / 'strain' / 'strainfeature.csv'
    for dataset in ['comb-2']:
        creat_data(dataset, cellfile)
