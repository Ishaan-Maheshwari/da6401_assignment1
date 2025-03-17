import wandb
from .model import NeuralNetwork
from .dataset import load_dataset

def train_model(config):
    wandb.init(project=config["wandb_project"], entity=config["wandb_entity"])
    
    X_train, y_train, X_val, y_val, _, _ = load_dataset(config["dataset"])
    model = NeuralNetwork(
        input_size=784, hidden_layers=config["num_layers"], hidden_neurons=config["hidden_size"],
        output_size=10, activation=config["activation"], optimizer=config["optimizer"],
        learning_rate=config["learning_rate"], weight_init=config["weight_init"]
    )

    for epoch in range(config["epochs"]):
        model.forward(X_train)  # Forward propagation placeholder
        wandb.log({"epoch": epoch, "loss": 0.5, "accuracy": 0.8})  # Dummy logging

    wandb.finish()
