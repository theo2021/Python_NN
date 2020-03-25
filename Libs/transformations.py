import numpy as np


class RbfTransformation:
    def __init__(self, mean_array, sigma_array):
        self.means = mean_array
        self.sigmas = sigma_array

    def __call__(self, inp):
        ans = np.zeros(len(self.means))
        for i, (mean, sigma) in enumerate(zip(self.means, self.sigmas)):
            ans[i] = np.exp(-(np.inner(inp - mean, inp - mean))/sigma/2)
        return ans
        #return np.exp(-(np.inner(inp - self.means, inp - self.means))/self.sigmas/2)

    def train_mu(self, data, step, epochs):
        for epoch in range(epochs):
            for sample in data[np.random.choice(len(data), len(data))]:
            #for sample in data:
                distances = np.zeros(len(self.means))
                for i in range(len(self.means)):
                    distances[i] = np.linalg.norm(sample-self.means[i])**2
                best = np.argmin(distances)
                self.means[best] += step * (sample - self.means[best])

    def train_mu_fsca(self, data, step, epochs):
        winners = np.ones(len(self.means))
        for epoch in range(epochs):
            for sample in data[np.random.choice(len(data), len(data))]:
                distances = np.zeros(len(self.means))
                for i in range(len(self.means)):
                    distances[i] = np.linalg.norm(sample - self.means[i])*winners[i]
                best = np.argmin(distances)
                self.means[best] += step * (sample - self.means[best])
                winners[best]+=1


class StepTransformation:
    def __init__(self, mean_array):
        self.means = mean_array

    def __call__(self, inp):
        ans = np.zeros(len(self.means))
        ans[np.argmin(np.abs(inp-self.means))] = 1
        return ans
