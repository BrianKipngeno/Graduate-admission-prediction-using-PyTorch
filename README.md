# Graduate-admission-prediction-using-PyTorch

This project aims to predict a student's chance of getting into graduate school based on several factors such as their GRE score, TOEFL score, university rating, and more. We use PyTorch to build and train a neural network for regression, with the goal of predicting the chance of admission as a continuous value.

**Project Overview**

Graduate admission prediction is a regression task that estimates the likelihood of a student getting accepted into a graduate program based on their academic and test performance. This project demonstrates:

- Preparing and processing the dataset.
- Building a neural network using PyTorch.
- Training the model and evaluating its performance.
- Making predictions and comparing them with actual data.
  
**Dataset**

The dataset includes the following features:

- GRE Score
- TOEFL Score
- University Rating
- Statement of Purpose (SOP) Score
- Letter of Recommendation (LOR) Score
- Undergraduate GPA
- Research Experience (binary)
- The target variable is the admit_chance, which is the probability of a student being admitted to a graduate program.

You can access the dataset here:  http://bit.ly/uni_admission

**Key Steps**

**1. Data Preparation**

Load the dataset using Pandas.

Drop unnecessary columns.

Split the dataset into features (inputs) and target (output).

Convert the data into PyTorch tensors for use in model training and evaluation.

**2. Model Building**

We build a multi-layer fully connected neural network using PyTorch's torch.nn.Module. The network architecture consists of seven layers. The input layer corresponds to the seven features, and the output is a single value representing the predicted chance of admission.

**3. Model Training**

The training loop is set up with the Mean Squared Error (MSE) loss function to measure the error between predicted and actual values. The Adam optimizer is used for gradient descent. The model is trained over 1000 epochs, with the loss being logged and monitored.

**4. Model Evaluation**

The performance of the model is evaluated using:

- Mean Absolute Error (MAE): Measures the average absolute difference between predicted and actual values.
- Mean Squared Error (MSE): Measures the average squared difference, which gives more weight to larger errors.
- 
These metrics are used to assess the accuracy of the model.

**5. Loss Visualization**

The loss values during training are plotted against the number of epochs to visualize how the model improves over time.

**6. Making Predictions**

The trained model predicts admission chances for students in the test set. We calculate the MAE and MSE to evaluate model performance on unseen data. Additionally, we compare the actual and predicted chances for the first 10 students in the test set.

**Results**

- Mean Absolute Error (MAE): 0.07
- Mean Squared Error (MSE): 0.01
- 
The model demonstrates good performance with low error values, indicating that it can accurately predict a student's chance of admission to a graduate program based on their test scores and other academic information.

**Conclusion**

This project illustrates how to effectively use PyTorch to build, train, and evaluate a neural network model for regression tasks. The model successfully predicts admission chances with reasonable accuracy, offering insights into how various factors influence the likelihood of getting into graduate school.

