import numpy as np


def deconstruct_matrix(V, W, dif=1e-5):
    H = np.zeros((W.shape[1], V.shape[1]))
    for x in range(W.shape[1]):
        selected_W = W[:, x]
        H[x] = np.mean(
            np.divide(V[selected_W != 0], W[:, x][selected_W != 0].reshape((-1, 1))),
            axis=0).reshape((-1, 1)).transpose()
    # cur_w = np.ones_like(W)

    top = np.matmul(W.transpose(), V)
    bottom_part_1 = np.matmul(W.transpose(), W)
    for x in range(3000):
        bottom = np.matmul(bottom_part_1, H)
        H_next = np.divide(np.multiply(H, top), bottom)
        if np.max(np.abs(H_next - H)) <= dif:
            print(x)
            break
        H = H_next
    print(3000)
    return H_next


if __name__ == '__main__':
    T = 15
    n = 10
    # c_b = np.array([[1,2,3,4],[4,3,2,1]]).reshape((4,2))#np.random.random((T,1))
    # a_b = np.array([[.5,3],[2.5,5],[7,9]]).reshape((2,3))#np.random.random((1,n))
    c_b = np.random.random([100, 35])
    a_b = np.random.random([35, 40])
    V = np.matmul(c_b, a_b)  # +np.random.normal(0, .1, (4,3))
    deconstruct_matrix(V, c_b)
