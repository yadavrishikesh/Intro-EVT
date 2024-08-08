rm(list = ls())  # Clear all objects from the workspace
setwd(this.path::here())  # Set the working directory to the directory containing the current script

# Load necessary libraries
library(ggplot2)  # For plotting
library(gridExtra)  # For arranging multiple plots in a grid

# Define the loss function (Mean Squared Error for polynomial regression)
loss_function_poly <- function(w, data, degree) {
  y_pred <- predict_poly(data$x, w, degree)  # Predict y using polynomial function
  sum((data$y - y_pred)^2) / nrow(data)  # Compute mean squared error
}

# Define the gradient of the loss function for polynomial regression
gradient_poly <- function(w, data, degree) {
  y_pred <- predict_poly(data$x, w, degree)  # Predict y using polynomial function
  gradients <- sapply(0:degree, function(j) {
    -2 * sum((data$y - y_pred) * (data$x^j)) / nrow(data)  # Compute gradient for each coefficient
  })
  return(gradients)  # Return gradients
}

# Polynomial prediction function
predict_poly <- function(x, w, degree) {
  y_pred <- sapply(0:degree, function(j) w[j+1] * x^j)  # Compute polynomial terms
  rowSums(y_pred)  # Sum the terms to get the prediction
}

# Gradient descent algorithm for polynomial regression
gradient_descent_poly <- function(data, learning_rate, iterations, degree) {
  w <- rep(0, degree + 1)  # Initialize coefficients to zero
  history <- data.frame(iteration = integer(iterations), matrix(nrow = iterations, ncol = degree + 1), loss = numeric(iterations))  # Create a data frame to store history
  
  for (i in 1:iterations) {
    grads <- gradient_poly(w, data, degree)  # Compute gradients
    w <- w - learning_rate * grads  # Update coefficients
    loss <- loss_function_poly(w, data, degree)  # Compute loss
    history[i, ] <- c(i, w, loss)  # Store iteration, coefficients, and loss
  }
  
  colnames(history)[2:(degree + 2)] <- paste0("w", 0:degree)  # Set column names for coefficients
  return(history)  # Return the history
}

# Generate synthetic data
set.seed(42)  # Set seed for reproducibility
n <- 1000  # Number of data points
x <- rnorm(n)  # Generate random normal data for x
w0 = 1; w1 = -1; w2=2.1;  w3=-2  # Define true coefficients
y <- w0 + w1 * x + w2 * x^2 + w3 * x^3 + rnorm(n, sd= sqrt(3))  # Generate y with polynomial relationship and noise
data <- data.frame(x = x, y = y)  # Create data frame

# Perform gradient descent for polynomial regression
degree <- 3  # Degree of polynomial
learning_rate <- 0.01  # Learning rate for gradient descent
iterations <- 1000  # Number of iterations for gradient descent
history <- gradient_descent_poly(data, learning_rate, iterations, degree)  # Perform gradient descent

# Plot the data points and the final regression curve
data$y_pred <- predict_poly(data$x, as.numeric(history[iterations, 2:(degree + 2)]), degree)  # Predict y using final coefficients
p1 <- ggplot(data, aes(x = x, y = y)) +
  geom_point(color = 'blue') +
  geom_line(aes(y = y_pred), color = 'red') +
  labs(title = "Data Points and Final Polynomial Regression Curve") +
  theme(plot.title = element_text(size = 20), axis.title.x = element_text(size = 15), axis.title.y = element_text(size = 15))

# Plot the loss function over iterations
p2 <- ggplot(history, aes(x = iteration, y = loss)) +
  geom_line(color = 'blue') +
  labs(title = "Loss Function Over Iterations") +
  theme(plot.title = element_text(size = 20), axis.title.x = element_text(size = 15), axis.title.y = element_text(size = 15))

# Plot the convergence of the polynomial coefficients
p3 <- ggplot(history, aes(x = iteration)) +
  geom_line(aes(y = w0), color = 'green') +
  geom_line(aes(y = w1), color = 'purple') +
  geom_line(aes(y = w2), color = 'orange') +
  geom_line(aes(y = w3), color = 'blue') +
  geom_hline(yintercept = w0, linetype = "dashed", color = "green") +
  geom_hline(yintercept = w1, linetype = "dashed", color = "purple") +
  geom_hline(yintercept = w2, linetype = "dashed", color = "orange") +
  geom_hline(yintercept = w3, linetype = "dashed", color = "blue") +
  labs(title = "Convergence of Polynomial Coefficients", y = "Coefficients") +
  theme(plot.title = element_text(size = 20), axis.title.x = element_text(size = 15), axis.title.y = element_text(size = 15)) +
  scale_color_manual(values = c("w0" = "green", "w1" = "purple", "w2" = "orange", "w3" = "blue"))

# Save the plots to a PDF file
pdf("../../RKeras/figures/gradient_descent_polynomial_harmonic_plots.pdf", width = 10, height = 12)  # Create PDF file for plots
grid.arrange(p1, p2, p3, nrow = 3)  # Arrange plots in a grid
dev.off()  # Close the PDF device
