import os
import json
import argparse
from rdkit import Chem

from ddc_pub import ddc_v3 as ddc


def load_model():
    path = os.path.join("model/heteroencoder_model")
    print("Loading heteroencoder model from:", path)
    model = ddc.DDC(model_name=path)
    return model


def encode(smiles_file, output_latent_file_path=None):

    model = load_model()

    smiles_in = []
    with open(smiles_file, "r") as file:
        smiles_in = json.load(file)

    mols_in = [
        Chem.rdchem.Mol.ToBinary(Chem.MolFromSmiles(smiles)) for smiles in smiles_in
    ]
    latent = model.transform(model.vectorize(mols_in))

    os.makedirs(os.path.dirname(output_latent_file_path), exist_ok=True)
    with open(output_latent_file_path, "w") as f:
        json.dump(latent.tolist(), f)

    print("Encoding completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and train a model")

    parser.add_argument("--smiles-file", type=str, required=True)
    parser.add_argument("--output_latent_file_path", type=str)
    args = {k: v for k, v in vars(parser.parse_args()).items() if v is not None}
    encode(**args)
