import sys
import argparse
import json
import torch
import numpy as np

from models import Generator


class Sampler(object):
    def __init__(self, generator: Generator):
        self.set_generator(generator)

    def set_generator(self, generator):
        self.G = generator

    def sample(self, n):
        z = torch.cuda.FloatTensor(np.random.uniform(-1, 1, (n, self.G.latent_dim)))
        return self.G(z)


class SampleModelRunner:
    def __init__(self, output_latent_file, input_model_path, sample_number):

        self.output_latent_file = output_latent_file
        self.input_model_path = input_model_path
        self.sample_number = sample_number

        self.G = Generator.load(input_model_path)

        if torch.cuda.is_available():
            self.G.cuda()
        self.Tensor = (
            torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        )

    def run(self):

        with torch.no_grad():
            self.G.eval()

            S = Sampler(generator=self.G)
            latent = S.sample(self.sample_number)
            latent = latent.detach().cpu().numpy().tolist()

            with open(self.output_latent_file, "w") as json_file:
                json.dump(latent, json_file)


def sample(
    generator_path,
    output_sampled_latent_file,
    number_samples,
):
    print("Sampling the generator")
    print(generator_path)

    print("Sampling model")
    S = SampleModelRunner(output_sampled_latent_file, generator_path, number_samples)
    S.run()

    print("Sampling finished\n")
    sys.stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and train a model")

    parser.add_argument("--generator_path", type=str, required=True)
    parser.add_argument("--output_sampled_latent_file", type=str, required=True)
    parser.add_argument("--number_samples", type=int)

    args = {k: v for k, v in vars(parser.parse_args()).items() if v is not None}
    sample(**args)
