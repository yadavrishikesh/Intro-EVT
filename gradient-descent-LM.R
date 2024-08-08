# Clear the workspace
rm(list = ls())  # Remove all objects from the current environment to start fresh

setwd(this.path::here())  # Set the working directory to the directory containing the current script

# Load necessary libraries
library(ggplot2)  # For creating plots
library(gridExtra)  # For arranging multiple plots in a grid layout

# Define the loss function (Mean Squared Error for linear regression)
loss_function_linear <- function(w, data) {
  y_pred <- predict_linear(data$x, w)  # Predict y using the linear model
  sum((data$y - y_pred)^2) / nrow(data)  # Compute and return the Mean Squared Error
}

# Define the gradient of the loss function for linear regression
gradient_linear <- function(w, data) {
  y_pred <- predict_linear(data$x, w)  # Predict y using the linear model
  gradients <- c(
    -2 * sum(data$y - y_pred) / nrow(data),  # Gradient for the intercept
    -2 * sum((data$y - y_pred) * data$x) / nrow(data)  # Gradient for the slope
  )
  return(gradients)  # Return the gradients
}

# Linear prediction function
predict_linear <- function(x, w) {
  y_pred <- w[1] + w[2] * x  # Compute the predicted y using the linear model
  return(y_pred)  # Return the predicted y
}

# Gradient descent algorithm for linear regression
gradient_descent_linear <- function(data, learning_rate, iterations) {
  w <- c(0, 0)  # Initialize coefficients to zero
  history <- data.frame(iteration = integer(iterations), intercept = numeric(iterations), slope = numeric(iterations), loss = numeric(iterations))  # Create a data frame to store the history of coefficients and loss
  
  for (i in 1:iterations) {
    grads <- gradient_linear(w, data)  # Compute gradients
    w <- w - learning_rate * grads  # Update coefficients using gradient descent
    loss <- loss_function_linear(w, data)  # Compute loss
    history[i, ] <- c(i, w[1], w[2], loss)  # Store iteration, coefficients, and loss
  }
  
  return(history)  # Return the history of the gradient descent process
}

# Generate synthetic data
set.seed(42)  # Set seed for reproducibility
n <- 1000  # Number of data points
x <- rnorm(n)  # Generate random normal data for x
y <- 2 + 3 * x + rnorm(n)  # Generate y with a linear relationship and added noise
data <- data.frame(x = x, y = y)  # Create a data frame with x and y

# Perform gradient descent for linear regression
learning_rate <- 0.01  # Learning rate for gradient descent
iterations <- 1000  # Number of iterations for gradient descent
history <- gradient_descent_linear(data, learning_rate, iterations)  # Perform gradient descent

# Plot the data points and the final regression line
data$y_pred <- predict_linear(data$x, as.numeric(history[iterations, 2:3]))  # Predict y using the final coefficients from gradient descent
p1 <- ggplot(data, aes(x = x, y = y)) +
  geom_point(color = 'blue') +  # Plot data points
  geom_line(aes(y = y_pred), color = 'red') +  # Plot the regression line
  labs(title = "Data Points and Final Linear Regression Line")  # Add a title to the plot

# Plot the loss function over iterations
p2 <- ggplot(history, aes(x = iteration, y = loss)) +
  geom_line(color = 'blue') +  # Plot the loss over iterations
  labs(title = "Loss Function Over Iterations")  # Add a title to the plot

# Plot the convergence of the intercept
p3 <- ggplot(history, aes(x = iteration, y = intercept)) +
  geom_line(color = 'green') +  # Plot the convergence of the intercept
  geom_hline(yintercept = 2, linetype = "dashed", color = "green") +  # Add a dashed line at the true value of the intercept
  labs(title = "Convergence of Intercept", y = "Intercept")  # Add a title and y-axis label to the plot

# Plot the convergence of the slope
p4 <- ggplot(history, aes(x = iteration, y = slope)) +
  geom_line(color = 'purple') +  # Plot the convergence of the slope
  geom_hline(yintercept = 3, linetype = "dashed", color = "purple") +  # Add a dashed line at the true value of the slope
  labs(title = "Convergence of Slope", y = "Slope")  # Add a title and y-axis label to the plot

# Save the plots to a PDF file
pdf("gradient_descent_linear_plots.pdf", width = 12, height = 12)  # Create a PDF file to save the plots
grid.arrange(p1, p2, p3, p4, nrow = 2, ncol = 2)  # Arrange the four plots in a 2x2 grid layout
dev.off()  # Close the PDF device
