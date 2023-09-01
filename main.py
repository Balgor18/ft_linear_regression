import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.datasets import make_regression

matplotlib.use('MacOSX')

FILENAME="data.csv"
DEBUG=0

# Modele part
def model(X, theta):
        return X.dot(theta)

# Cout
def cost_function(X, y, theta):
     m = len(y)
     return 1/(2*m) * np.sum((model(X, theta) - y)**2)

#Gradient
def grad(X, y, theta):
    m = len(y)
    return 1/m * X.T.dot(model(X, theta) - y)


def gradient_descent(X, y, theta, learn, n_iter):
    for i in range (0, n_iter):
        theta = theta - learn * grad(X, y, theta)
        if (DEBUG):
            print(str(i) + " ---> "+ str(n_iter))
    return theta

# def gradient_descent(X, y, theta, learning_rate, n_iterations):
    
#     cost_history = np.zeros(n_iterations) # création d'un tableau de stockage pour enregistrer l'évolution du Cout du modele
    
#     for i in range(0, n_iterations):
#         theta = theta - learning_rate * grad(X, y, theta) # mise a jour du parametre theta (formule du gradient descent)
#         cost_history[i] = cost_function(X, y, theta) # on enregistre la valeur du Cout au tour i dans cost_history[i]
        
#     return theta, cost_history


def main():
    '''
        Entry point of the programm
    '''
    
    # data = np.loadtxt(FILENAME, delimiter=',', skiprows=1)

    # Km
    # x = data[:, 0]
    # x = x.reshape(x.shape[0],1)
    # # Price
    # y = data[:, 1]
    # y = y.reshape(y.shape[0],1)


    x, y = make_regression(n_samples=100, n_features=1, noise=10)

    if (DEBUG) :
        print(x.shape)
        print(y.shape)


    X = np.hstack((x, np.ones(x.shape)))
    print(X.shape)
    
    np.random.seed(0)
    theta = np.random.randn(2, 1)
    # theta = [1000, 1000]

    model(X, theta)

    n_iterations = 1000
    learning_rate = 0.01


    # theta_final, cost_history = gradient_descent(X, y, theta, learning_rate, n_iterations)
    theta_final = gradient_descent(X, y, theta, learn=0.0001, n_iter=1000)
    print(cost_function(X, y, theta_final))
    # print(theta_final)

    prediction = model(X, theta_final)
    plt.scatter(x, y)
    plt.plot(x, prediction, c='r')
    plt.show()
    # cost = cost_function(X, y, theta);
    # print(cost)
    # print(theta_final)
    exit(2)
    plt.scatter(x, y)
    plt.plot(x, model(X, theta_final), c='r')
    plt.show()
    # plt.savefig('test.png')


if __name__ == '__main__':
    main()