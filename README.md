# **Feedforward Neural Network Training & Hyperparameter Sweeping**  

This repository provides a **custom feedforward neural network** built from scratch and allows:  
- Training the model using `train.py`  
- Running **automated hyperparameter tuning** using `sweep.py`  
- Logging results to **Weights & Biases (wandb)**  

---

## **📌 Index**
1. [Installation & Setup](#installation--setup)  
2. [Project Structure](#project-structure)  
3. [How to Train the Model (`train.py`)](#how-to-train-the-model-trainpy)  
4. [How to Run a Hyperparameter Sweep (`sweep.py`)](#how-to-run-a-hyperparameter-sweep-sweeppy)  
5. [GitHub Repository](#github-repository)
6. [WandB Report](#wandb-report)


## **1️⃣ Installation & Setup**  

### **🔹 Install Required Dependencies**  
Make sure you have Python **3.7+** installed. Then, install required packages:  

```bash
pip install -r requirements.txt
```

### **🔹 Set Up Weights & Biases (wandb)**  
Before running the training or sweep, log in to **Weights & Biases**:  
```bash
wandb login
```
If you don’t have a WandB account, create one at [https://wandb.ai](https://wandb.ai).

---

# **2️⃣ Project Structure**  

The project follows a **modular structure**, making it easy to maintain, expand, and reuse different components. Each module is responsible for a specific task, such as dataset handling, model creation, optimizers, training, and hyperparameter tuning.

---

## **📂 Root Directory Overview**
```
/root_directory
│── train.py        # Main script to train the neural network
│── sweep.py        # Script to run a WandB hyperparameter sweep
│── my_nn/          # Neural network module (contains all core implementations)
│   ├── __init__.py
│   ├── dataset.py
│   ├── activations.py
│   ├── initializers.py
│   ├── optimizers.py
│   ├── model.py
│   ├── trainer.py
│── README.md       # Documentation on how to use the project
```
---

## **📂 Detailed Breakdown of Each Component**

### **1️⃣ `train.py` – Training Script**
📌 **Purpose:**  
- Allows users to train the neural network with **custom hyperparameters**.
- Accepts command-line arguments to modify model settings.
- Logs training results to **Weights & Biases (wandb)**.

📌 **Key Features:**  
✔ Parses command-line arguments (epochs, batch size, optimizer, etc.).  
✔ Loads dataset and initializes the neural network.  
✔ Runs training loop and logs performance.  
✔ Handles errors (invalid hyperparameters, WandB connection issues, etc.).  

---

### **2️⃣ `sweep.py` – Hyperparameter Tuning Script**
📌 **Purpose:**  
- Runs a **WandB sweep**, automatically training the model with different hyperparameters.
- Helps find the best settings for **optimal accuracy and loss**.

📌 **Key Features:**  
✔ Defines a sweep **configuration** (ranges for hyperparameters).  
✔ Runs multiple **experiments** with different hyperparameter combinations.  
✔ Logs results to **WandB**, making it easy to analyze best-performing settings.  

---

### **3️⃣ `my_nn/` – Neural Network Package**
This directory contains the core implementation of the **feedforward neural network**.

---

#### **📂 `my_nn/__init__.py` – Package Initialization**
📌 **Purpose:**  
- Initializes the **my_nn** package, making it easy to import different components.  
- Defines which modules can be accessed directly.  


---

#### **📂 `my_nn/dataset.py` – Dataset Handling**
📌 **Purpose:**  
- Loads **MNIST** or **Fashion-MNIST** datasets.  
- Preprocesses data (normalization, reshaping, one-hot encoding).  
- Splits dataset into **train, validation, and test sets**.  


---

#### **📂 `my_nn/activations.py` – Activation Functions**
📌 **Purpose:**  
- Implements **activation functions** and their derivatives for backpropagation.  

📌 **Supported Activations:**  
✔ Sigmoid  
✔ Tanh  
✔ ReLU  
✔ Identity (for linear layers)  



---

#### **📂 `my_nn/initializers.py` – Weight Initialization**
📌 **Purpose:**  
- Defines different **weight initialization methods** to improve model performance.  

📌 **Supported Methods:**  
✔ **Random Initialization** (Small Gaussian noise)  
✔ **Xavier Initialization** (For deep networks, prevents exploding/vanishing gradients)  


---

#### **📂 `my_nn/optimizers.py` – Optimization Algorithms**
📌 **Purpose:**  
- Implements different **optimizers** for training the model.  

📌 **Supported Optimizers:**  
✔ **SGD (Stochastic Gradient Descent)**  
✔ **Momentum (Improves SGD convergence)**  
✔ **NAG (Nesterov Accelerated Gradient)**  
✔ **RMSprop (Adaptive learning rate optimizer)**  
✔ **Adam (Most commonly used optimizer)**  
✔ **Nadam (Adam + Nesterov Momentum)**  


---

#### **📂 `my_nn/model.py` – Neural Network Model**
📌 **Purpose:**  
- Implements **forward and backward propagation**.  
- Supports **custom architectures** with multiple hidden layers.  
- Uses **different activation functions & optimizers**.  

📌 **Key Methods:**  
✔ `forward(X)`: Compute activations layer-by-layer.  
✔ `backward(X, y_true, activations)`: Compute gradients for backpropagation.  
✔ `update_weights(grads)`: Apply optimizer updates.  


---

#### **📂 `my_nn/trainer.py` – Training Logic**
📌 **Purpose:**  
- Runs **training loop** and updates weights using selected optimizer.  
- Logs training results to **WandB**.  

📌 **Key Features:**  
✔ Runs multiple **epochs**, updates weights in **mini-batches**.  
✔ Computes **loss & accuracy** on training and validation sets.  
✔ Logs everything in **Weights & Biases (wandb)**.  

---

## **3️⃣ How to Train the Model (`train.py`)**  

The `train.py` script **trains the neural network** using custom hyperparameters and logs results to **WandB**.

### **🔹 Basic Usage**
```bash
python train.py --wandb_entity myname --wandb_project myprojectname
```
This will train the model using **default hyperparameters**.

### **🔹 Customizing Hyperparameters**
You can change hyperparameters by passing them as command-line arguments:
```bash
python train.py --epochs 10 --batch_size 16 --optimizer adam --learning_rate 0.001
```

### **🔹 Available Arguments**
| Argument | Default | Description |
|----------|---------|-------------|
| `-wp`, `--wandb_project` | `myprojectname` | Name of the WandB project |
| `-we`, `--wandb_entity` | `myname` | WandB entity name |
| `-d`, `--dataset` | `fashion_mnist` | Dataset (`mnist` or `fashion_mnist`) |
| `-e`, `--epochs` | `1` | Number of training epochs |
| `-b`, `--batch_size` | `4` | Batch size |
| `-l`, `--loss` | `cross_entropy` | Loss function (`mean_squared_error` or `cross_entropy`) |
| `-o`, `--optimizer` | `sgd` | Optimizer (`sgd`, `adam`, `nadam`, etc.) |
| `-lr`, `--learning_rate` | `0.1` | Learning rate |
| `-m`, `--momentum` | `0.5` | Momentum for `momentum` and `nag` optimizers |
| `-beta`, `--beta` | `0.5` | Beta for `rmsprop` |
| `-beta1`, `--beta1` | `0.5` | Beta1 for `adam` and `nadam` |
| `-beta2`, `--beta2` | `0.5` | Beta2 for `adam` and `nadam` |
| `-eps`, `--epsilon` | `1e-6` | Epsilon for numerical stability |
| `-w_d`, `--weight_decay` | `0.0` | Weight decay |
| `-w_i`, `--weight_init` | `random` | Weight initialization (`random` or `Xavier`) |
| `-nhl`, `--num_layers` | `1` | Number of hidden layers |
| `-sz`, `--hidden_size` | `4` | Neurons per hidden layer |
| `-a`, `--activation` | `sigmoid` | Activation function (`identity`, `sigmoid`, `tanh`, `ReLU`) |

---

## **4️⃣ How to Run a Hyperparameter Sweep (`sweep.py`)**  

The `sweep.py` script **automatically searches for the best hyperparameters** using **WandB sweeps**.

### **🔹 Basic Usage**
```bash
python sweep.py --wandb_entity myname --wandb_project myprojectname --count 10
```
This runs **10 different training experiments** with random hyperparameter values.

### **🔹 Available Arguments**
| Argument | Default | Description |
|----------|---------|-------------|
| `-wp`, `--wandb_project` | `myprojectname` | Name of the WandB project |
| `-we`, `--wandb_entity` | `myname` | WandB entity name |
| `-c`, `--count` | `10` | Number of sweep runs |

### **🔹 What Happens in a Sweep?**
1. `sweep.py` initializes a **WandB sweep** with predefined hyperparameter ranges.
2. It runs multiple **training experiments**, each with a different set of hyperparameters.
3. Logs results to WandB, helping you find the **best-performing configuration**.

---

## **5️⃣ GitHub Repository**

You can find the GitHub repository for this project [here](https://github.com/Ishaan-Maheshwari/da6401_assignment1.git).

---

## **6️⃣ WandB Report**
The link to the W&B report is: [https://wandb.ai/ishaan_maheshwari-indian-institute-of-technology-madras/assignment-1/reports/Ishaan-Maheshwari-s-DA6401-Assignment-1--VmlldzoxMTgyMzQwNQ?accessToken=qmqrmp6mxgsqt3tzvrm5yh7fubh94l108hn6qqtvxo5epipaekwlxa6lzvbk282e](https://wandb.ai/ishaan_maheshwari-indian-institute-of-technology-madras/assignment-1/reports/Ishaan-Maheshwari-s-DA6401-Assignment-1--VmlldzoxMTgyMzQwNQ?accessToken=qmqrmp6mxgsqt3tzvrm5yh7fubh94l108hn6qqtvxo5epipaekwlxa6lzvbk282e)

---
