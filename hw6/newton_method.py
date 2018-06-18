import numpy as np

def hessian(x):
    return 4 / (np.exp(2*x) + np.exp(-2*x) + 2)

def gradient(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def newton_decrement(x):
    return np.sqrt(gradient(x) * (1 / hessian(x)) * gradient(x))

def f(x):
    return np.log(np.exp(x) + np.exp(-x))

def newton_method(x0, epsilon=1e-4):
    x = x0
    t = 1
    counter = 0
    max_loop = 20
    print("x0: {}".format(x0))
    while True:
        print("x: {}".format(x))
        print("f(x): {}".format(f(x)))
        delta_x_nt = - (1 / hessian(x)) * gradient(x)
        # Stopping criterion
        if (newton_decrement(x) ** 2) / 2 <= epsilon:
            break
        # Update
        x = x + delta_x_nt
        counter += 1
        if counter > max_loop:
            break

    print("=" * 20)


def main():
    newton_method(1)
    newton_method(1.1)

if __name__ == '__main__':
    main()