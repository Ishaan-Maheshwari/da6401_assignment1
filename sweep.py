import wandb
import argparse
from my_nn.trainer import train_model

# Define the sweep configuration
sweep_config = {
    "method": "bayes",  # Options: "grid", "random", "bayes"
    "metric": {"name": "val_loss", "goal": "minimize"},
    "parameters": {
        "epochs": {"values": [10, 20, 30]},
        "batch_size": {"values": [16, 32, 64]},
        "learning_rate": {"values": [0.001, 0.01, 0.1]},
        "optimizer": {"values": ["sgd", "adam", "nadam"]},
        "momentum": {"values": [0.5, 0.9]},
        "beta1": {"values": [0.9, 0.99]},
        "beta2": {"values": [0.999, 0.9999]},
        "epsilon": {"values": [1e-6, 1e-8]},
        "weight_decay": {"values": [0.0, 0.01]},
        "num_layers": {"values": [1, 2, 3]},
        "hidden_size": {"values": [32, 64, 128]},
        "activation": {"values": ["sigmoid", "tanh", "ReLU"]},
        "weight_init": {"values": ["random", "Xavier"]},
    }
}

# Sweep function
def train_sweep():
    """ Train the model using the hyperparameters provided by wandb sweep. """
    wandb.init()
    config = wandb.config  # Get hyperparameters from wandb

    # Convert wandb config to dictionary format required by train_model
    config_dict = {
        "wandb_project": wandb.run.project,
        "wandb_entity": wandb.run.entity,
        "dataset": "fashion_mnist",
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "optimizer": config.optimizer,
        "momentum": config.momentum,
        "beta1": config.beta1,
        "beta2": config.beta2,
        "epsilon": config.epsilon,
        "weight_decay": config.weight_decay,
        "num_layers": config.num_layers,
        "hidden_size": config.hidden_size,
        "activation": config.activation,
        "weight_init": config.weight_init,
        "loss": "cross_entropy",  # Fixed loss function
    }

    train_model(config_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a WandB Sweep for Hyperparameter Tuning.")
    parser.add_argument("-wp", "--wandb_project", default="myprojectname", help="WandB project name")
    parser.add_argument("-we", "--wandb_entity", default="myname", help="WandB entity")
    parser.add_argument("-c", "--count", type=int, default=10, help="Number of sweep runs")

    args = parser.parse_args()

    # Initialize WandB sweep
    wandb.login()
    sweep_id = wandb.sweep(sweep_config, project=args.wandb_project, entity=args.wandb_entity)

    # Run the sweep
    wandb.agent(sweep_id, function=train_sweep, count=args.count)
