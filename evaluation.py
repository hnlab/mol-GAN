import json
import moses
import rdkit
import numpy as np
import pandas as pd
from tqdm import trange, tqdm
from rdkit import Chem, rdBase
from rdkit.Chem import RDConfig, Descriptors, rdMolDescriptors, AllChem, Draw
from rdkit.Chem.Draw.MolDrawing import MolDrawing, DrawingOptions

rdBase.DisableLog("rdApp.*")

import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
# now you can import sascore!
import sascorer
import argparse


def moses_evaluation(gen_smi_filepath, train_smi_filepath, moses_metrics_path, valid_k):

    print(gen_smi_filepath)

    # Timing function
    def task_begin():
        import time

        since = time.time()
        print("The task is being executed...\n----------")
        return since

    def task_done(since):
        import time

        print("----------\nThe task has been done.")
        time_elapsed = time.time() - since
        print("Time cost {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

    # Get train & generated data
    with open(train_smi_filepath, "r") as f:
        train_smi = json.load(f)
    print(f"The number of trainning smiles is {len(train_smi)}.")

    with open(gen_smi_filepath, "r") as f:
        gen_smi = json.load(f)
    print(f"The number of genetated smiles is {len(gen_smi)}.")

    # Moses evaluation
    since = task_begin()
    metrics = moses.get_all_metrics(
        gen_smi,
        k=valid_k,
        n_jobs=1,
        device="cuda:0",
        batch_size=512,
        pool=None,
        test=train_smi,
        test_scaffolds=train_smi,
        ptest=None,
        ptest_scaffolds=None,
        train=train_smi,
    )
    task_done(since)

    with open(moses_metrics_path, "w") as f:
        json.dump(metrics, f)


def load_and_process_property(gen_smi_filepath, property_savepath):

    # Test smi validity
    def valid_or_not(smi):
        if len(smi) == 0:
            return False
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return False
        try:
            Descriptors.qed(mol)
        except Exception as e:
            print(e)
            return False
        return True

    def compute_prop(df):
        # There are 21 molecular descriptors

        df["MolWt"] = df.mol.map(Descriptors.MolWt)
        df["MolLogP"] = df.mol.map(Descriptors.MolLogP)
        df["BertzCT"] = df.mol.map(Descriptors.MolLogP)
        df["TPSA"] = df.mol.map(Descriptors.TPSA)
        df["MolMR"] = df.mol.map(Descriptors.MolMR)

        df["FractionCSP3"] = df.mol.map(Descriptors.FractionCSP3)
        df["NumHDonors"] = df.mol.map(Descriptors.NumHDonors)
        df["NumHAcceptors"] = df.mol.map(Descriptors.NumHAcceptors)
        df["NumRotatableBonds"] = df.mol.map(Descriptors.NumRotatableBonds)
        df["NumHeteroatoms"] = df.mol.map(Descriptors.NumHeteroatoms)

        df["HeavyAtomCount"] = df.mol.map(Descriptors.HeavyAtomCount)
        df["RingCount"] = df.mol.map(Descriptors.RingCount)
        df["NumAliphaticRings"] = df.mol.map(Descriptors.NumAliphaticRings)
        df["NumAromaticRings"] = df.mol.map(Descriptors.NumAromaticRings)
        df["NumSaturatedRings"] = df.mol.map(Descriptors.NumSaturatedRings)

        df["NumValenceElectrons"] = df.mol.map(Descriptors.NumValenceElectrons)
        df["NumAmideBonds"] = df.mol.map(rdMolDescriptors.CalcNumAmideBonds)
        df["NumBridgeheadAtoms"] = df.mol.map(rdMolDescriptors.CalcNumBridgeheadAtoms)
        df["NumSpiroAtoms"] = df.mol.map(rdMolDescriptors.CalcNumSpiroAtoms)
        df["qed"] = df.mol.map(Descriptors.qed)

        df["SA"] = df.mol.map(sascorer.calculateScore)

    rdBase.DisableLog("rdApp.*")

    with open(gen_smi_filepath, "r") as f:
        all_smi = json.load(f)

    smi_list = [i for i in all_smi if valid_or_not(i)]
    df = pd.DataFrame(smi_list, columns=["SMILES"])
    df["mol"] = [Chem.MolFromSmiles(smi) for smi in tqdm(df["SMILES"])]

    compute_prop(df)
    df.to_csv(property_savepath, index=False)

    rdBase.EnableLog("rdApp.*")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="moses evaluation and property computation"
    )

    parser.add_argument("--gen_smi_filepath", type=str, required=True)
    parser.add_argument("--train_smi_filepath", type=str)
    parser.add_argument("--valid_k", type=int)
    parser.add_argument("--property_savepath", type=str)
    parser.add_argument("--moses_metrics_path", type=str)

    args = {k: v for k, v in vars(parser.parse_args()).items() if v is not None}

    moses_evaluation(
        gen_smi_filepath=args["gen_smi_filepath"],
        train_smi_filepath=args["train_smi_filepath"],
        moses_metrics_path=args["moses_metrics_path"],
        valid_k=args["valid_k"],
    )
    load_and_process_property(
        gen_smi_filepath=args["gen_smi_filepath"],
        property_savepath=args["property_savepath"],
    )
