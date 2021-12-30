import numpy as np
from tqdm import tqdm
import cv2
import os

import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.tensorboard import SummaryWriter

from PIL import Image

class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()        
        # N, 1, 160, 120
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 3, stride=2, padding=1), # -> N, 16, 80,  60
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, 3, stride=2, padding=1), # -> N, 32, 40, 30
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, stride=2, padding=1), # -> N, 32, 20, 15
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 15) # -> N, 64, 6, 1
        )
        
        # N , 64, 1, 1
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128, 64, 15), # -> N, 32, 7, 7
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), # N, 16, 14, 14 (N,16,13,13 without output_padding)
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), # N, 16, 14, 14 (N,16,13,13 without output_padding)
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), # N, 1, 28, 28  (N,1,27,27)
            torch.nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
class NeuralNetworkPolicy:
    """
    A wrapper to train neural network model
    ...
    Methods
    -------
    optimize(observations, expert_actions, episode)
        train the model on the newly collected data from the simulator

    predict(observation)
        takes images and predicts the action space using the model

    save
        save a model checkpoint to storage location
    """

    def __init__(self, model, optimizer, storage_location, dataset, **kwargs):
        """
        Parameters
        ----------
        model :
            input pytorch model that will be trained
        optimizer :
            torch optimizer
        storage_location : string
            path of the model to be saved , the dataset and tensorboard logs
        dataset :
            object storing observation and labels from expert
        """
        self._train_iteration = 0
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Base parameters
        self.model = model.to(self._device)

        self.model_autoencoder = AE().to(self._device)
        self.optimizer_autoencoder = torch.optim.Adam(self.model_autoencoder.parameters(),
                             lr = 1e-3, weight_decay = 1e-5)
        self.loss_function_auto = torch.nn.MSELoss()

        self.optimizer = optimizer
        self.storage_location = storage_location
        self.writer = SummaryWriter(self.storage_location)

        # Optional parameters
        self.epochs = kwargs.get("epochs", 10)
        self.batch_size = kwargs.get("batch_size", 32)
        self.input_shape = kwargs.get("input_shape", (60, 80))
        self.max_velocity = kwargs.get("max_velocity", 0.7)

        self.episode = 0

        self.dataset = dataset

        self.dataset_autoencoder = []
        # Load previous weights
        if "model_path" in kwargs:
            self.model.load_state_dict(torch.load(kwargs.get("model_path"), map_location=self._device))
            print("Loaded ")

    def __del__(self):
        self.writer.close()

    def optimize(self, observations, expert_actions, episode):
        """
        Parameters
        ----------
        observations :
            input images collected from the simulator
        expert_actions :
            list of actions each action [velocity, steering_angle] from expert
        episode : int
            current episode number
        """
        print("Starting episode #", str(episode))
        original_observations = observations
        self.episode = episode
        self.model.episode = episode
        # Transform newly received data
        observations, expert_actions = self._transform(observations, expert_actions)

        # Retrieve data loader
        dataloader = self._get_dataloader(observations, expert_actions)
        # Train model
        for epoch in tqdm(range(1, self.epochs + 1)):
            running_loss = 0.0
            self.model.epoch = epoch
            for i, data in enumerate(dataloader, 0):
                # Send data to device
                data = [variable.to(self._device) for variable in data]

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                loss = self.model.loss(*data)
                loss.backward()
                self.optimizer.step()

                # Statistics
                running_loss += loss.item()

            # Logging
            self._logging(loss=running_loss, epoch=epoch)


        '''Autoencoder training'''

        print("Training autoencoder")

        optimizer = self.optimizer_autoencoder
        
        # Validation using MSE Loss function

        observations_auto = self._transform_autoencoder(original_observations)

        dataloader_auto = self._get_dataloader_autoencoder(observations_auto)

        epochs = 5#0
        losses_epochs = []
        
        for epoch in range(epochs):
            print("Epoch", epoch)
            losses = []
            for image in dataloader_auto:
                
            # Output of Autoencoder
                image = image.to(self._device)
                reconstructed = self.model_autoencoder(image)
                
            # Calculating the loss function
                loss = self.loss_function_auto(reconstructed, image)
                #print(loss.item())
                
            # The gradients are set to zero,
            # the the gradient is computed and stored.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            # Storing the losses in a list for plotting
                losses.append(loss.item())
            loss_mean = sum(losses)/len(losses)
            print("Mean loss", loss_mean)
            losses_epochs.append(loss_mean)


        print("Last epoch autoencoder mean loss", losses_epochs[-1])

        # Post training routine
        self._on_optimization_end()

    def predict(self, observation):
        """
        Parameters
        ----------
        observations :
            input image from the simulator
        Returns
        ----------
        prediction:
            action space of input observation
        """
        # Apply transformations to data
        observation, _ = self._transform([observation], [0])
        observation = torch.tensor(observation)
        # Predict with model
        prediction = self.model.predict(observation.to(self._device))

        return prediction


    def autoencoder_predict(self, observation):

        observation = self._transform_autoencoder([observation])
        observation = torch.tensor(observation)
        # Predict with model
        observation = observation.to(self._device)
        reconstructed = self.model_autoencoder(observation)

        loss = self.loss_function_auto(reconstructed, observation)


        return loss


    def save(self):
        torch.save(self.model.state_dict(), os.path.join(self.storage_location, "model.pt"))
        torch.save(self.model_autoencoder.state_dict(), "autoencoder.pt")

    def _transform(self, observations, expert_actions):
        # Resize images
        observations = [
            Image.fromarray(cv2.resize(observation, dsize=self.input_shape[::-1]))
            for observation in observations
        ]
        # Transform to tensors
        compose_obs = Compose(
            [
                ToTensor(),
                Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),  # using imagenet normalization values
            ]
        )

        observations = [compose_obs(observation).cpu().numpy() for observation in observations]
        try:
            # Scaling steering angle to become in range -1 to 1 to make it easier to regress
            # Scaling velocity to range 0-1 based on max velocity
            expert_actions = [
                np.array([expert_action[0] / self.max_velocity, expert_action[1] / (np.pi / 2)])
                for expert_action in expert_actions
            ]
        except:
            pass
        expert_actions = [torch.tensor(expert_action).cpu().numpy() for expert_action in expert_actions]

        return observations, expert_actions

    def _transform_autoencoder(self, observations):
        input_shape = (160, 120) # (80, 60)
        observations = ([cv2.resize(observation, dsize=input_shape)
            for observation in observations])
        observations_gr = [Image.fromarray(cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY) )for observation in observations]
        compose_obs_gr = Compose([
                ToTensor(),
                Normalize(
                (0.5), (0.5)
                ), ])

        observations_gr = [compose_obs_gr(observation).cpu().numpy() for observation in observations_gr]
        return observations_gr
#        pass
    
    def _get_dataloader_autoencoder(self, observations):
        batch_size = 8  
        self.dataset_autoencoder.extend(observations)
        dataloader = DataLoader(self.dataset_autoencoder, batch_size=batch_size, shuffle=True)

        return dataloader

    def _get_dataloader(self, observations, expert_actions):
        # Include new experiences
        self.dataset.extend(observations, expert_actions)
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        return dataloader

    def _logging(self, **kwargs):
        epoch = kwargs.get("epoch")
        loss = kwargs.get("loss")
        self.writer.add_scalar("Loss/train/{}".format(self._train_iteration), loss, epoch)

    def _on_optimization_end(self):
        self._train_iteration += 1
