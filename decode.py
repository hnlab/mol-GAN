import json
import moses
import os, sys
import argparse
import numpy as np
import pandas as pd
from tqdm import trange
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw.MolDrawing import MolDrawing, DrawingOptions

from ddc_pub import ddc_v3 as ddc

rdBase.DisableLog("rdApp.*")


def load_model():
    path = os.path.join("model/heteroencoder_model")
    print("Loading heteroencoder model from:", path)
    model = ddc.DDC(model_name=path)
    return model


# Validity
def valid_or_not(smi):
    if len(smi) == 0:
        return False
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False
    else:
        return True


def latent_decode(latent_filepath, smi_decoded_filepath):

    print(latent_filepath)

    # Read latent data
    with open(latent_filepath, "r") as f:
        sampled_latent = json.load(f)

    # Load model to decode
    model = load_model()

    # Decode
    invalids = 0
    # decoding batch size
    batch_size = 256
    n = len(sampled_latent)
    smile_decoded = []

    for indx in range(0, n // batch_size):
        lat = np.array(sampled_latent[(indx) * 256 : (indx + 1) * 256])
        if indx % 10 == 0:
            print(
                "Batch [%d/%d] decoded, [Invalids: %s]"
                % (indx, n // batch_size + 1, invalids)
            )
            sys.stdout.flush()

        try:
            smiles, _ = model.predict_batch(lat, temp=0)
        except Exception as e:
            print(e)
            continue

        smile_decoded += list(smiles)
        for smi in smiles:
            if not valid_or_not(smi):
                invalids += 1

    print(
        "All latent number is {}, Last decode index is {}".format(
            len(sampled_latent), (indx + 1) * 256
        )
    )

    latent_remain = np.array(sampled_latent[(indx + 1) * 256 :]).reshape(-1, 1, 512)

    smi_addition = []
    for lat in latent_remain:
        try:
            smiles, _ = model.predict(lat, temp=0)
        except Exception as e:
            print(e)
            continue
        smi_addition.append(smiles)
        if not valid_or_not(smiles):
            invalids += 1

    smile_decoded += smi_addition
    print("Decoding completed.")
    num_smiles = len(smile_decoded)
    print(
        "Total: {} Fraction Valid: {:.2%}\n\n".format(
            num_smiles, (num_smiles - invalids) / num_smiles
        )
    )

    rdBase.EnableLog("rdApp.*")

    # save smi_decoded
    with open(smi_decoded_filepath, "w") as f:
        json.dump(smile_decoded, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and train a model")

    parser.add_argument("--latent_filepath", type=str, required=True)
    parser.add_argument("--smi_decoded_filepath", type=str)

    args = {k: v for k, v in vars(parser.parse_args()).items() if v is not None}
    latent_decode(**args)
