from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import numpy as np

X, Y = make_classification(n_classes=2, n_samples=400, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=3)

Y[Y == 0] = -1


class SVM:
    def __init__(self, C=1.0):
        self.C = C
        self.W = 0
        self.b = 0

    def hingeloss(self, W, b, X, Y):
        loss = 0
        loss += 0.5 * np.dot(W.T, W)

        m = X.shape[0]
        for i in range(m):
            ti = Y[i] * (np.dot(W, X[i].T) + b)
            loss += self.C * max(0, (1 - ti))
        return loss[0][0]

    def fit(self, X, Y, batch_size=100, learning_rate=0.001, maxItr=300):
        no_of_features = X.shape[1]
        no_of_samples = X.shape[0]
        n = learning_rate
        c = self.C

        W = np.zeros((1, no_of_features))
        bias = 0

        losses = []
        for i in range(maxItr):
            l = self.hingeloss(W, bias, X, Y)
            losses.append(l)
            ids = np.arange(no_of_samples)
            np.random.shuffle(ids)
            for batch_start in range(0, no_of_samples, batch_size):
                gradw = 0
                gradb = 0
                for j in range(batch_start, batch_size + batch_size):
                    if j < no_of_samples:
                        i = ids[j]
                        ti = Y[i] * (np.dot(W, X[i].T) + bias)
                        if ti > 1:
                            gradw += 0
                            gradb += 0
                        else:
                            gradw += c * Y[i] * X[i]
                            gradb += c * Y[i]
                W = W * (1 - n) + n * gradw
                bias = bias + n * gradb
        self.W = W
        self.b = bias
        return W, bias, losses


def plotHyperplane(w1, w2, b):
    x1 = np.linspace(-2, 4, 10)
    x2 = -(w1 * x1 + b) / w2
    plt.plot(x1, x2)
    plt.scatter(X[:, 0], X[:, 1], c=Y)
    plt.show()


mysvm = SVM()
W, b, losses = mysvm.fit(X, Y)

plotHyperplane(W[0, 0], W[0, 1], b)
