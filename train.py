import argparse
import wandb
import sys
from my_nn.trainer import train_model

def validate_args(args):
    """ Validate command-line arguments before starting training. """
    if args.epochs <= 0:
        raise ValueError("Number of epochs must be greater than 0.")
    if args.batch_size <= 0:
        raise ValueError("Batch size must be greater than 0.")
    if args.learning_rate <= 0:
        raise ValueError("Learning rate must be greater than 0.")
    if args.momentum < 0 or args.momentum > 1:
        raise ValueError("Momentum should be in the range [0, 1].")
    if args.beta < 0 or args.beta > 1:
        raise ValueError("Beta should be in the range [0, 1].")
    if args.beta1 < 0 or args.beta1 > 1:
        raise ValueError("Beta1 should be in the range [0, 1].")
    if args.beta2 < 0 or args.beta2 > 1:
        raise ValueError("Beta2 should be in the range [0, 1].")
    if args.epsilon <= 0:
        raise ValueError("Epsilon must be greater than 0.")
    if args.weight_decay < 0:
        raise ValueError("Weight decay must be non-negative.")
    if args.num_layers < 1:
        raise ValueError("Number of hidden layers must be at least 1.")
    if args.hidden_size < 1:
        raise ValueError("Hidden layer size must be at least 1.")

def main():
    """ Main function to parse arguments and start training. """
    parser = argparse.ArgumentParser(description="Train a feedforward neural network.")

    # WandB arguments
    parser.add_argument("-wp", "--wandb_project", default="assignment-1", help="WandB project name")
    parser.add_argument("-we", "--wandb_entity", default="ishaan_maheshwari-indian-institute-of-technology-madras", help="WandB entity")

    # Dataset & training parameters
    parser.add_argument("-d", "--dataset", choices=["mnist", "fashion_mnist"], default="fashion_mnist", help="Dataset to use")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=4, help="Batch size")

    # Loss and optimizer parameters
    parser.add_argument("-l", "--loss", choices=["mean_squared_error", "cross_entropy"], default="cross_entropy", help="Loss function")
    parser.add_argument("-o", "--optimizer", choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], default="sgd", help="Optimizer")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.1, help="Learning rate")
    parser.add_argument("-m", "--momentum", type=float, default=0.5, help="Momentum for momentum/NAG optimizers")
    parser.add_argument("-beta", "--beta", type=float, default=0.5, help="Beta for RMSprop")
    parser.add_argument("-beta1", "--beta1", type=float, default=0.5, help="Beta1 for Adam/Nadam")
    parser.add_argument("-beta2", "--beta2", type=float, default=0.5, help="Beta2 for Adam/Nadam")
    parser.add_argument("-eps", "--epsilon", type=float, default=1e-6, help="Epsilon for numerical stability")
    parser.add_argument("-w_d", "--weight_decay", type=float, default=0.0, help="Weight decay for optimizers")

    # Model architecture
    parser.add_argument("-w_i", "--weight_init", choices=["random", "Xavier"], default="random", help="Weight initialization method")
    parser.add_argument("-nhl", "--num_layers", type=int, default=1, help="Number of hidden layers")
    parser.add_argument("-sz", "--hidden_size", type=int, default=4, help="Number of neurons per hidden layer")
    parser.add_argument("-a", "--activation", choices=["identity", "sigmoid", "tanh", "ReLU"], default="sigmoid", help="Activation function")

    try:
        args = parser.parse_args()

        # Validate parsed arguments
        validate_args(args)

        # Convert parsed arguments to dictionary
        config = vars(args)

        # Ensure WandB login works before proceeding
        try:
            wandb.login()
        except Exception as e:
            print(f"❌ Error: Unable to login to WandB. Check your API key.\n{e}")
            sys.exit(1)

        # Start training
        train_model(config)

    except ValueError as ve:
        print(f"❌ Argument Error: {ve}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
