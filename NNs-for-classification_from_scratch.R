rm(list = ls())
setwd(this.path::here())
# Load necessary libraries
library(ggplot2)  # For plotting

# Generate synthetic data
set.seed(42)  # For reproducibility
n <- 1000  # Number of data points
x1 <- rnorm(n)  # Random normal data for x1
x2 <- rnorm(n)  # Random normal data for x2
# Non-linear relationship with some noise
y <- ifelse(x1^2 + x2^2 + rnorm(n) > 1.5, 1, 0)
data <- data.frame(x1 = x1, x2 = x2, y = y)

# Visualize the data
ggsave(file= "../../RKeras/figures/classification_examples.pdf", width = 10, height = 8)
ggplot(data, aes(x = x1, y = x2, color = factor(y))) +
  geom_point() +
  labs(title = "Synthetic Data", x = "x1", y = "x2") +
  scale_color_manual(values = c("red", "blue")) + 
  theme(
    plot.title = element_text(size = 20, face = "bold"),
    axis.title.x = element_text(size = 15),
    axis.title.y = element_text(size = 15)
  )


# Split data into training and testing sets
set.seed(42)  # For reproducibility
train_indices <- sample(1:n, size = 0.7 * n)  # 70% for training
train_data <- data[train_indices, ]
test_data <- data[-train_indices, ]

# Activation function (sigmoid)
sigmoid <- function(z) {
  1 / (1 + exp(-z))  # Sigmoid activation function
}

# Derivative of sigmoid function
sigmoid_derivative <- function(z) {
  sigmoid(z) * (1 - sigmoid(z))  # Derivative of sigmoid for back propagation
}

# Forward propagation
forward_propagation <- function(X, W1, b1, W2, b2) {
  Z1 <- X %*% W1 + matrix(b1, nrow = nrow(X), ncol = ncol(W1), byrow = TRUE)  # First layer linear combination
  A1 <- sigmoid(Z1)  # First layer activation
  Z2 <- A1 %*% W2 + matrix(b2, nrow = nrow(A1), ncol = ncol(W2), byrow = TRUE)  # Second layer linear combination
  A2 <- sigmoid(Z2)  # Second layer activation
  list(Z1 = Z1, A1 = A1, Z2 = Z2, A2 = A2)  # Return intermediate values for backpropagation
}

# Compute loss (mean squared error)
compute_loss <- function(Y, A2) {
  sum((Y - A2)^2) / length(Y)  # Mean squared error
}

# Backward propagation
backward_propagation <- function(X, Y, W1, b1, W2, b2, Z1, A1, Z2, A2) {
  m <- nrow(X)  # Number of samples
  dZ2 <- A2 - Y  # Error in output layer
  dW2 <- t(A1) %*% dZ2 / m  # Gradient for W2
  db2 <- colSums(dZ2) / m  # Gradient for b2
  dZ1 <- dZ2 %*% t(W2) * sigmoid_derivative(Z1)  # Error in hidden layer
  dW1 <- t(X) %*% dZ1 / m  # Gradient for W1
  db1 <- colSums(dZ1) / m  # Gradient for b1
  list(dW1 = dW1, db1 = db1, dW2 = dW2, db2 = db2)  # Return gradients
}

# Training the neural network
train_neural_network <- function(X, Y, hidden_units, learning_rate, iterations) {
  input_units <- ncol(X)  # Number of input features
  output_units <- 1  # Single output unit (binary classification)
  
  # Initialize weights and biases with random values
  W1 <- matrix(rnorm(input_units * hidden_units), nrow = input_units, ncol = hidden_units)
  b1 <- rep(0, hidden_units)
  W2 <- matrix(rnorm(hidden_units * output_units), nrow = hidden_units, ncol = output_units)
  b2 <- rep(0, output_units)
  
  # Store loss history for plotting
  loss_history <- numeric(iterations)
  
  for (i in 1:iterations) {
    # Forward propagation
    forward <- forward_propagation(X, W1, b1, W2, b2)
    Z1 <- forward$Z1
    A1 <- forward$A1
    Z2 <- forward$Z2
    A2 <- forward$A2
    
    # Compute loss
    loss <- compute_loss(Y, A2)
    loss_history[i] <- loss
    
    # Backward propagation
    backward <- backward_propagation(X, Y, W1, b1, W2, b2, Z1, A1, Z2, A2)
    dW1 <- backward$dW1
    db1 <- backward$db1
    dW2 <- backward$dW2
    db2 <- backward$db2
    
    # Update weights and biases
    W1 <- W1 - learning_rate * dW1
    b1 <- b1 - learning_rate * db1
    W2 <- W2 - learning_rate * dW2
    b2 <- b2 - learning_rate * db2
    
    if (i %% 100 == 0) {
      cat("Iteration:", i, "Loss:", loss, "\n")  # Print loss every 100 iterations
    }
  }
  
  list(W1 = W1, b1 = b1, W2 = W2, b2 = b2, loss_history = loss_history)
}

# Prepare the input data and target
X <- as.matrix(train_data[, c("x1", "x2")])  # Input features
Y <- as.matrix(train_data$y)  # Target values

# Train the neural network
set.seed(42)  # For reproducibility
hidden_units <- 10  # Number of hidden units
learning_rate <- 0.1  # Learning rate for gradient descent
iterations <- 50000  # Number of iterations for training
model <- train_neural_network(X, Y, hidden_units, learning_rate, iterations)

# Plot loss over iterations
loss_history <- data.frame(iteration = 1:iterations, loss = model$loss_history)
p1<- ggplot(loss_history, aes(x = iteration, y = loss)) +
  geom_line(color = 'blue') +
  labs(title = "Loss Function Over Iterations", x = "Iteration", y = "Loss") +
  theme(
    plot.title = element_text(size = 20, face = "bold"),
    axis.title.x = element_text(size = 15),
    axis.title.y = element_text(size = 15)
  )

# Make predictions on the test set
test_X <- as.matrix(test_data[, c("x1", "x2")])
forward_test <- forward_propagation(test_X, model$W1, model$b1, model$W2, model$b2)
test_predictions <- ifelse(forward_test$A2 > 0.5, 1, 0)

# Evaluate the model
confusion_matrix <- table(test_data$y, test_predictions)
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
print(confusion_matrix)
print(paste("Accuracy:", accuracy))

# Plot the decision boundary
plot_data <- expand.grid(x1 = seq(min(data$x1), max(data$x1), length.out = 200),
                         x2 = seq(min(data$x2), max(data$x2), length.out = 200))
forward_plot <- forward_propagation(as.matrix(plot_data), model$W1, model$b1, model$W2, model$b2)
plot_data$y <- as.numeric(forward_plot$A2)

p2<- ggplot(data, aes(x = x1, y = x2, color = factor(y))) +
  geom_point() +
  geom_contour(data = plot_data, aes(z = y), breaks = 0.5, color = "red") +
  labs(title = "Decision Boundary of the Neural Network", x = "x1", y = "x2") +
  theme(
    plot.title = element_text(size = 20, face = "bold"),
    axis.title.x = element_text(size = 15),
    axis.title.y = element_text(size = 15)
  )

pdf(file= "../../RKeras/figures/classification_examples_resulst.pdf", width = 10, height = 6)
gridExtra::grid.arrange(p1, p2, ncol=2)
dev.off()
