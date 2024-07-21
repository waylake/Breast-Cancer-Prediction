- `README.md`: Project documentation.
- `data/wdbc.data`: Breast Cancer Wisconsin dataset.
- `data/wdbc.names`: Description of the dataset.
- `overfitted_prediction.py`: Neural network implementation prone to overfitting.
- `prediction.py`: Neural network implementation with techniques to mitigate overfitting.

## How to Run

1. Clone the repository.
2. Navigate to the project directory.
3. Ensure you have the required dependencies installed.
4. Run `prediction.py` to train the model with techniques to mitigate overfitting and evaluate its performance.
5. Run `overfitted_prediction.py` to train the model prone to overfitting and evaluate its performance.

```bash
git clone <repository-url>
cd Breast-Cancer-Prediction
pip install numpy pandas scikit-learn
python prediction.py
python overfitted_prediction.py
```

## Dataset 
The dataset used is the Breast Cancer Wisconsin dataset, which is included in the `data` directory. This dataset contains various features of cell nuclei from breast cancer biopsies. 
- **ID** : Sample identifier.
 
- **Diagnosis** : The diagnosis of breast cancer (M = malignant, B = benign).

- Various features of cell nuclei: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension, each measured for the mean, standard error, and worst/largest.

## Description 

### overfitted_prediction.py 

This script contains the implementation of a neural network that is prone to overfitting.
 
- **Data Loading and Preprocessing** : The dataset is loaded and preprocessed, including label encoding and feature scaling.
 
- **Neural Network Class** : A simple neural network class with one hidden layer is defined.
 
- **Training** : The network is trained for a specified number of epochs without techniques to mitigate overfitting.
 
- **Evaluation** : The model is evaluated on the test set.

### prediction.py 

This script contains the implementation of a neural network with techniques to mitigate overfitting.
 
- **Data Loading and Preprocessing** : The dataset is loaded and preprocessed, including label encoding and feature scaling.
 
- **Neural Network Class** : A neural network class with one hidden layer and dropout regularization is defined.
 
- **Training** : The network is trained with dropout regularization and early stopping to prevent overfitting.
 
- **Evaluation** : The model is evaluated on the test set.

### Key Techniques to Mitigate Overfitting 
 
- **Dropout Regularization** : Randomly drops units from the neural network during training to prevent over-reliance on specific paths.
 
- **Early Stopping** : Monitors validation loss and stops training when the validation loss stops improving, preventing the model from overfitting the training data.

## Results 
The performance of both models is evaluated using accuracy on the test set. `prediction.py` incorporates techniques to mitigate overfitting and is expected to generalize better to new data compared to `overfitted_prediction.py`.
### Example Output 
**overfitted_prediction.py**

```bash
Epoch 0, Loss: 0.3373452911520941, Validation Loss: 0.4037526750869745
...
Early stopping due to no improvement in validation loss
Accuracy: 0.98
```
**prediction.py**

```bash
Epoch 0, Loss: 0.3373452911520941, Validation Loss: 0.2037526750869745
...
Accuracy: 0.60
```

## Dependencies 

- numpy

- pandas

- scikit-learn

You can install the dependencies using the following command:


```bash
pip install numpy pandas scikit-learn
```

## License 

This project is licensed under the MIT License.


```bash
This README file provides a comprehensive guide to the project, including its structure, how to run the scripts, a description of the dataset, and an explanation of the code and techniques used to mitigate overfitting.
```
