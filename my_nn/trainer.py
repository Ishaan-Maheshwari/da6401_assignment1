import wandb
from my_nn.dataset import load_dataset
from my_nn.model import NeuralNetwork

def train_model(config):
    """ Train the neural network with real wandb logging. """
    
    # Initialize wandb
    wandb.init(project=config["wandb_project"], entity=config["wandb_entity"])

    run_name = (f"OPT-{config['optimizer'].upper()}_LR-{config['learning_rate']}_"
                f"LAYERS-{config['num_layers']}_NEURONS-{config['hidden_size']}_"
                f"ACT-{config['activation'].upper()}_BS-{config['batch_size']}")
    
    wandb.run.name = run_name 
    wandb.run.save()
    
    # Load dataset
    X_train, y_train, X_val, y_val, _, _ = load_dataset(config["dataset"])
    
    # Initialize model
    model = NeuralNetwork(
        input_size=784,
        hidden_layers=config["num_layers"],
        hidden_neurons=config["hidden_size"],
        output_size=10,
        activation=config["activation"],
        optimizer=config["optimizer"],
        learning_rate=config["learning_rate"],
        weight_init=config["weight_init"],
        loss_function=config["loss"]
    )

    # Training loop with real logging
    for epoch in range(config["epochs"]):
        train_loss = model.train(X_train, y_train, batch_size=config["batch_size"])
        val_loss, val_acc = model.evaluate(X_val, y_val)

        # Log real metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

        print(f"Epoch {epoch+1}/{config['epochs']} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")

    wandb.finish()
