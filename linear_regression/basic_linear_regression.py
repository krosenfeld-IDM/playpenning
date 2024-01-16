import os
import numpy as np
import matplotlib.pyplot as plt

# change working directory to the current file's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def part_0():
    """Generating synthetic data"""

    np.random.seed(0)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)

    # Plotting the generated data
    plt.figure()
    plt.scatter(X, y, label='Data')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.title('Synthetic Data')
    plt.savefig('0_synthetic_data.png')

    return X, y


def part_1(X, y):
    """Linear regression using the normal equation"""

    # Add a column of ones to X to account for the intercept
    X_b = np.c_[np.ones((len(X), 1)), X]    

    # Solving the least squares problem directly using the normal equation
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    # Predictions using the obtained parameters
    y_pred = X_b.dot(theta_best)

    # Calculating metrics
    bias = np.mean(y_pred) - np.mean(y)
    SSE = np.sum(np.square(y_pred - y))
    MSE = np.mean(np.square(y_pred - y))
    MAE = np.mean(np.abs(y_pred - y))
    RMSE = np.sqrt(MSE)

    # Results
    print('----------------------------------------------------')
    print('Part 1: Linear regression using the normal equation')
    print(f'Predicted parameters: {theta_best.flatten()}')
    print(f'bias: {bias:.4f}')
    print(f'SSE: {SSE:.4f}')
    print(f'MSE: {MSE:.4f}')
    print(f'MAE: {MAE:.4f}')
    print(f'RMSE: {RMSE:.4f}')

    # Plotting the generated data
    plt.figure()
    plt.scatter(X, y, label='Data')
    plt.plot(plt.gca().get_xlim(), theta_best[0] + theta_best[1] * plt.gca().get_xlim(), 
            'k', label='Normal equation')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Part 1: Linear regression\nusing the normal equation')
    plt.savefig('1_normal_equation.png')

    return theta_best


def part_2(X, y, theta_compare=None):
    """Linear regression using gradient descent"""

    # Add a column of ones to X to account for the intercept
    X_b = np.c_[np.ones((len(X), 1)), X]    

    # Function to compute the cost (sum of squared residuals)
    def compute_cost(X, y, theta):
        m = len(y)
        predictions = X.dot(theta)
        cost = (1/2*m) * np.sum(np.square(predictions - y))
        return cost

    # Gradient Descent Function
    def gradient_descent(X, y, theta, learning_rate, iterations):
        m = len(y)
        cost_history = np.zeros(iterations)
        
        for i in range(iterations):
            predictions = X.dot(theta)
            theta = theta - (1/m) * learning_rate * (X.T.dot((predictions - y)))
            cost_history[i] = compute_cost(X, y, theta)
        
        return theta, cost_history

    # Initial parameters (theta), learning rate and number of iterations
    theta = np.zeros((2, 1))
    learning_rate = 0.01
    iterations = 1000

    # Running Gradient Descent
    theta_final, cost_history = gradient_descent(X_b, y, theta, learning_rate, iterations)

    # Predictions using the obtained parameters
    y_pred = X_b.dot(theta_final)

    # Calculating metrics
    bias = np.mean(y_pred) - np.mean(y)
    SSE = np.sum(np.square(y_pred - y))
    MSE = np.mean(np.square(y_pred - y))
    MAE = np.mean(np.abs(y_pred - y))
    RMSE = np.sqrt(MSE)

    # Results
    print('----------------------------------------------------')
    print('Part 2: Linear regression using gradient descent')
    print(f'Predicted parameters: {theta_final.flatten()}')
    print(f'bias: {bias:.4f}')
    print(f'SSE: {SSE:.4f}')
    print(f'MSE: {MSE:.4f}')
    print(f'MAE: {MAE:.4f}')
    print(f'RMSE: {RMSE:.4f}')

    # Plotting the generated data
    plt.figure()
    plt.scatter(X, y, label='Data')
    if theta_compare is not None:
        plt.plot(plt.gca().get_xlim(), theta_compare[0] + theta_compare[1] * plt.gca().get_xlim(), 
                'k', label='Normal equation')
    plt.plot(plt.gca().get_xlim(), theta_final[0] + theta_final[1] * plt.gca().get_xlim(), 
            '-g', label='Gradient descent')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Part 2: Linear regression\nusing gradient descent')
    plt.savefig('2_gradient_descent.png')

    return

if __name__ == "__main__":
    X, y = part_0()
    theta_compare = part_1(X, y)
    part_2(X, y, theta_compare=theta_compare)