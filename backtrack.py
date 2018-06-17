import numpy as np

def gradient_descent():
    f = lambda x: np.sum(x ** 2) if x[0] > 1 else float("inf")
    gradient_f = lambda x: 2 * x
    
    x = np.array([2,2])
    

    alpha = 0.25
    beta = 0.5
    for _ in range(20):
        # Set delta x
        gradient = gradient_f(x)
        delta_x = -gradient
        # Find step size
        t = 1
        while f(x + t * delta_x) >= f(x) + alpha * t * np.dot(gradient, delta_x) :
            t = beta * t
        # Update
        x = x + t * delta_x

    return x

def main():
    final_x = gradient_descent()
    print(final_x)

if __name__ == '__main__':
    main()