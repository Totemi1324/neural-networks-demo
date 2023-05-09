# How to edit the neural network (regression)

<details>
<summary>TABLE OF CONTENTS</summary>

- [Editing the layers](#editing-the-layers)
- [The norm variable](#the-norm-variable)
- [Customize Dense layers](#customize-dense-layers)
- [The last Dense layer](#the-last-dense-layer)
- [Tuning the learning_rate variable](#tuning-the-learning_rate-variable)
- [Choosing a loss function](#choosing-a-loss-function)
</details>

---

To make changes to the neural network, you only need to edit the following code block:
```python
def build_and_compile_model(norm):
  model = Sequential([
      norm,
      Dense(64, activation='relu'),
      Dense(64, activation='relu'),
      Dense(1)
  ])

  learning_rate = 0.001

  model.compile(loss='mean_absolute_error',
                optimizer=Adam(learning_rate))
  return model
```
The following subsections offer guidance on the possibilities to tune the network architecture.

## Editing the layers

To edit the layers of the network, you can modify the layers inside the `Sequential()` function. The current architecture has three dense layers. You can add or remove dense layers to create a deeper or shallower network.

Here is an example of adding an extra dense layer with 32 neurons and 'linear' activation:
```python
model = Sequential([
    norm,
    Dense(64, activation='relu'),
    Dense(32, activation='linear'),
    Dense(64, activation='relu'),
    Dense(1)
])
```

## The `norm` variable

The `norm` variable should be left untouched. It is used to normalize the input features and is already defined elsewhere in the code.

## Customize `Dense` layers

`Dense` layers are fully connected hidden layers or, if the last in the list, output layers. Their size can be adjusted by modifying the number of neurons in the parentheses. The following activations are available:

- *relu*: Rectified Linear Unit activation, most commonly used activation function in deep learning
- *selu*: Scaled Exponential Linear Unit activation, useful for deeper networks
- *gelu*: Gaussian Error Linear Unit activation, similar to 'relu' but smoother
- *elu*: Exponential Linear Unit activation, similar to 'relu' but smoother and more robust to noise
- *sigmoid*: Sigmoid activation, used for binary classification problems
- *linear*: Linear activation, used for regression problems

Here is an example of replacing the 'relu' activations with 'selu' activations:

```python
model = Sequential([
    norm,
    Dense(64, activation='selu'),
    Dense(64, activation='selu'),
    Dense(1)
])
```

## The last `Dense` layer

The last `Dense` layer with a size of 1 is the output layer and should be left untouched.

## Tuning the `learning_rate` variable

The `learning_rate` variable determines how quickly the model learns from the data. A higher learning rate will result in faster learning, but can cause the model to overshoot the optimal weights and become unstable. A lower learning rate will result in slower learning, but the model will be more stable.

You can modify the `learning_rate` variable to fine-tune the model's performance. Here is an example of setting the learning rate to 0.0001:

```python
learning_rate = 0.0001
```

## Choosing a loss function

The loss function is used to evaluate how well the model is performing during training. In this notebook, there are two possible losses available:

- *mean_absolute_error*: Computes the mean absolute error between the predicted and actual values. This is a good choice for most regression problems.
- *mean_squared_error*: Computes the mean squared error between the predicted and actual values. This is a good choice for regression problems where larger errors should be penalized more heavily.

You can modify the loss function to better suit the problem you are trying to solve. Here is an example of changing the loss function to 'mean_squared_error':
```python
model.compile(loss='mean_squared_error',
            optimizer=Adam(learning_rate))
```