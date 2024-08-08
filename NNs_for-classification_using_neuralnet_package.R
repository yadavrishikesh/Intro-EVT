# Install and load necessary packages
library(neuralnet)
library(ggplot2)

# Generate synthetic data
set.seed(42)
n <- 1000
x1 <- rnorm(n)
x2 <- rnorm(n)
y <- ifelse(x1^2 + x2^2 + rnorm(n) > 1.5, 1, 0)
data <- data.frame(x1 = x1, x2 = x2, y = y)

# Split data into training and test sets
set.seed(42)
train_indices <- sample(1:n, size = 0.7 * n)
train_data <- data[train_indices, ]
test_data <- data[-train_indices, ]

# Define the neural network formula
formula <- y ~ x1 + x2

# Train the neural network
set.seed(42)
nn <- neuralnet(formula,
                data = train_data,
                hidden = c(10),          # 1 hidden layer with 10 neurons
                linear.output = FALSE,   # For classification problems
                stepmax = 1e6,           # Maximum number of steps
                learningrate = 0.1)      # Learning rate

# Plot the neural network architecture
plot(nn)

# Forward propagation
forward_propagation <- function(X, weights, biases, activation_function) {
  Z <- X %*% weights + biases
  A <- activation_function(Z)
  return(list(Z = Z, A = A))
}

# Activation function (sigmoid)
sigmoid <- function(z) {
  1 / (1 + exp(-z))
}

# Evaluate the model on test data
test_X <- as.matrix(test_data[, c("x1", "x2")])
predictions <- compute(nn, test_X)$net.result
predicted_classes <- ifelse(predictions > 0.5, 1, 0)

# Calculate accuracy
confusion_matrix <- table(test_data$y, predicted_classes)
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
print(confusion_matrix)
print(paste("Accuracy:", accuracy))

# Plot decision boundary
plot_data <- expand.grid(x1 = seq(min(data$x1), max(data$x1), length.out = 200),
                         x2 = seq(min(data$x2), max(data$x2), length.out = 200))
pred_grid <- compute(nn, as.matrix(plot_data))$net.result
plot_data$y <- ifelse(pred_grid > 0.5, 1, 0)

ggplot(data, aes(x = x1, y = x2, color = factor(y))) +
  geom_point() +
  geom_contour(data = plot_data, aes(z = y), breaks = 0.5, color = "red") +
  labs(title = "Decision Boundary of the Neural Network", x = "x1", y = "x2") +
  theme(
    plot.title = element_text(size = 20, face = "bold"),
    axis.title.x = element_text(size = 15),
    axis.title.y = element_text(size = 15)
  )
