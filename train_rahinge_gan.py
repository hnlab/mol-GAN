import os
import sys
import json
import time
import math
import torch
import pickle
import argparse
import warnings
import numpy as np
from tqdm import tqdm
from torch.utils import data
import torch.autograd as autograd

warnings.filterwarnings("ignore")

from models import Generator
from models import Discriminator
from encode import encode


class CreateModelRunner:
    def __init__(self, input_data_path, output_model_folder):

        self.input_data_path = input_data_path
        self.output_model_folder = output_model_folder

        with open(self.input_data_path, "r") as latent_vector_file:
            latent_space_mols = np.array(json.load(latent_vector_file))

        shape = latent_space_mols.shape
        assert len(shape) == 3
        self.data_shape = tuple([shape[1], shape[2]])

    def run(self):
        self.CreateGenerator()
        self.CreateDiscriminator()

    def CreateDiscriminator(self):

        D = Discriminator(self.data_shape)

        if not os.path.exists(self.output_model_folder):
            os.makedirs(self.output_model_folder)
        discriminator_path = os.path.join(self.output_model_folder, "discriminator.txt")
        D.save(discriminator_path)

    def CreateGenerator(self):

        G = Generator(self.data_shape, latent_dim=self.data_shape[1])

        if not os.path.exists(self.output_model_folder):
            os.makedirs(self.output_model_folder)
        generator_path = os.path.join(self.output_model_folder, "generator.txt")
        G.save(generator_path)


class Sampler(object):
    def __init__(self, generator: Generator):
        self.set_generator(generator)

    def set_generator(self, generator):
        self.G = generator

    def sample(self, n):
        z = torch.cuda.FloatTensor(np.random.uniform(-1, 1, (n, self.G.latent_dim)))
        return self.G(z)


class LatentMolsDataset(data.Dataset):
    def __init__(self, latent_space_mols):
        self.data = latent_space_mols

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class TrainModelRunner:
    def __init__(
        self,
        input_data_path,
        output_model_folder,
        n_epochs,
        starting_epoch,
        save_interval,
        warmup,
        lr_init,
        n_critic=5,
        batch_size=64,
        b1=0.5,
        b2=0.999,
        message="",
    ):

        self.input_data_path = input_data_path
        self.output_model_folder = output_model_folder
        self.n_epochs = n_epochs
        self.starting_epoch = starting_epoch
        self.save_interval = save_interval
        self.warmup = warmup
        self.lr_init = lr_init

        self.n_critic = n_critic
        self.batch_size = batch_size
        self.b1 = b1
        self.b2 = b2
        self.message = message

        with open(self.input_data_path, "r") as json_smiles:
            latent_space_mols = np.array(json.load(json_smiles))
        latent_space_mols = latent_space_mols.reshape(latent_space_mols.shape[0], 512)
        self.dataloader = torch.utils.data.DataLoader(
            LatentMolsDataset(latent_space_mols),
            shuffle=True,
            batch_size=self.batch_size,
            drop_last=True,
        )

        discriminator_name = (
            "discriminator.txt"
            if self.starting_epoch == 1
            else str(self.starting_epoch - 1) + "_discriminator.txt"
        )
        discriminator_path = os.path.join(output_model_folder, discriminator_name)
        self.D = Discriminator.load(discriminator_path)

        generator_name = (
            "generator.txt"
            if self.starting_epoch == 1
            else str(self.starting_epoch - 1) + "_generator.txt"
        )
        generator_path = os.path.join(output_model_folder, generator_name)
        self.G = Generator.load(generator_path)

        self.Sampler = Sampler(self.G)

        # Cosine learning rate
        t = self.warmup
        T = self.n_epochs + 1
        n_t = 0.5
        lr_rate = self.lr_init
        lambda1 = (
            lambda epoch: (0.9 * epoch / t + 0.1)
            if epoch < t
            else 0.1
            if n_t * (1 + math.cos(math.pi * (epoch - t) / (T - t))) < 0.1
            else n_t * (1 + math.cos(math.pi * (epoch - t) / (T - t)))
        )

        self.g_optimizer = torch.optim.AdamW(
            self.G.parameters(), lr=lr_rate, betas=(self.b1, self.b2)
        )
        self.g_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.g_optimizer, lr_lambda=lambda1
        )

        self.d_optimizer = torch.optim.AdamW(
            self.D.parameters(), lr=lr_rate, betas=(self.b1, self.b2)
        )
        self.d_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.d_optimizer, lr_lambda=lambda1
        )

        # Use GPU
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.G.cuda()
            self.D.cuda()
        self.Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    def lr_step(self):
        """Update learning rate."""
        self.g_scheduler.step()
        self.d_scheduler.step()

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def run(self):

        print("Training of Model started.")
        print("Message: {}".format(self.message))
        sys.stdout.flush()

        disc_loss_log = []
        g_loss_log = []

        for epoch in range(self.starting_epoch, self.n_epochs + self.starting_epoch):
            # Learning rate update
            self.lr_step()
            print(
                "Gen learning rate：[{:.6f}]".format(
                    self.g_optimizer.param_groups[0]["lr"]
                )
            )
            print(
                "Disc learning rate：[{:.6f}]".format(
                    self.d_optimizer.param_groups[0]["lr"]
                )
            )

            disc_loss_per_batch = []
            g_loss_log_per_batch = []

            for i, real_mols in enumerate(tqdm(self.dataloader)):

                real_mols = real_mols.type(self.Tensor)
                fake_mols = self.Sampler.sample(real_mols.shape[0]).type(self.Tensor)

                # ---------------------
                #  Train Discriminator
                # ---------------------

                real_pred = self.D(real_mols)
                fake_pred = self.D(fake_mols.detach())

                d_loss_1 = torch.mean(
                    torch.nn.ReLU()(1.0 - (real_pred - torch.mean(fake_pred)))
                )
                d_loss_2 = torch.mean(
                    torch.nn.ReLU()(1.0 + (fake_pred - torch.mean(real_pred)))
                )
                d_loss = (d_loss_1 + d_loss_2) / 2
                disc_loss_per_batch.append(d_loss.item())

                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

                if i % self.n_critic == 0:
                    # -----------------
                    #  Train Generator
                    # -----------------

                    fake_mols = self.Sampler.sample(real_mols.shape[0]).type(
                        self.Tensor
                    )

                    real_pred = self.D(real_mols)
                    fake_pred = self.D(fake_mols)

                    g_loss_1 = torch.mean(
                        torch.nn.ReLU()(1.0 + (real_pred - torch.mean(fake_pred)))
                    )
                    g_loss_2 = torch.mean(
                        torch.nn.ReLU()(1.0 - (fake_pred - torch.mean(real_pred)))
                    )
                    g_loss = (g_loss_1 + g_loss_2) / 2
                    g_loss_log_per_batch.append(g_loss.item())

                    self.reset_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                if i == len(self.dataloader) - 1:
                    if epoch % self.save_interval == 0:
                        generator_save_path = os.path.join(
                            self.output_model_folder, str(epoch) + "_generator.txt"
                        )
                        discriminator_save_path = os.path.join(
                            self.output_model_folder, str(epoch) + "_discriminator.txt"
                        )
                        self.G.save(generator_save_path)
                        self.D.save(discriminator_save_path)

                    disc_loss_log.append(
                        [time.time(), epoch, np.mean(disc_loss_per_batch)]
                    )
                    g_loss_log.append(
                        [time.time(), epoch, np.mean(g_loss_log_per_batch)]
                    )

                    print(
                        "[Epoch %d/%d]  [Disc loss: %.6f] [Gen loss: %.6f]"
                        % (
                            epoch,
                            self.n_epochs + self.starting_epoch,
                            disc_loss_log[-1][2],
                            g_loss_log[-1][2],
                        )
                    )
                    sys.stdout.flush()

        with open(
            os.path.join(self.output_model_folder, "disc_loss.json"), "w"
        ) as json_file:
            json.dump(disc_loss_log, json_file)
        with open(
            os.path.join(self.output_model_folder, "gen_loss.json"), "w"
        ) as json_file:
            json.dump(g_loss_log, json_file)


class RunRunner:
    def __init__(
        self,
        encode,
        smiles_file,
        storage_path,
        n_epochs,
        save_interval,
        starting_epoch,
        warmup,
        lr_init,
        latent_file="encoded_smiles.latent",
    ):

        self.enc = encode

        self.smiles_file = smiles_file
        self.storage_path = storage_path
        self.n_epochs = n_epochs
        self.save_interval = save_interval
        self.starting_epoch = starting_epoch
        self.warmup = warmup
        self.lr_init = lr_init

        self.output_latent = os.path.join(self.storage_path, latent_file)

    def run(self):
        print("Model Mol_wgan-gp running, encoding training set")
        if self.enc == "yes":
            encode(
                smiles_file=self.smiles_file, output_latent_file_path=self.output_latent
            )
        print("Encoding finished. Creating model files")
        C = CreateModelRunner(
            input_data_path=self.output_latent, output_model_folder=self.storage_path
        )
        C.run()
        print("Model Created. Training model")
        T = TrainModelRunner(
            input_data_path=self.output_latent,
            output_model_folder=self.storage_path,
            n_epochs=self.n_epochs,
            starting_epoch=self.starting_epoch,
            save_interval=self.save_interval,
            warmup=self.warmup,
            lr_init=self.lr_init,
        )
        T.run()
        print("Model finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and train a model")

    parser.add_argument("--encode", help="need encode or not [yes|no]", type=str)
    parser.add_argument(
        "--smiles_file", help="The path to a data file.[json]", type=str
    )
    parser.add_argument("--storage_path", help="The path to all outputs", type=str)
    parser.add_argument("--n_epochs", type=int, help="number of epochs of training")
    parser.add_argument("--starting_epoch", type=int, help="starting ofepoch")
    parser.add_argument("--save_interval", type=int, help="Save interval")
    parser.add_argument("--warmup", type=int, help="Cosine learning rate warmup")
    parser.add_argument("--lr_init", type=float, help="Initial learning rate")

    args = {k: v for k, v in vars(parser.parse_args()).items() if v is not None}
    runner = RunRunner(**args)
    runner.run()
