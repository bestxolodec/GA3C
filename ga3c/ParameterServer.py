import numpy as np
from time import sleep
from Database import RedisInterface


class Adam(object):

    def __init__(self, lr=0.001, beta=0.9, mu=0.999, eps=1e-8):
        self.lr, self.beta, self.mu, self.eps = lr, beta, mu, eps
        self.v = self.g = None
        self.t = 0

    def apply_grads(self, params, grads):
        if self.v is None: self.v = [p * 0 for p in params]
        if self.g is None: self.g = [p * 0 for p in params]
        self.t += 1
        updated_params = []
        for i, (param, grad) in enumerate(zip(params, grads)):
            self.v[i] = self.beta * self.v[i] + (1 - self.beta) * grad
            self.g[i] = self.mu * self.g[i] + (1 - self.mu) * grad ** 2
            v_normed = self.v[i] / (1 - self.beta ** self.t)
            g_normed = self.g[i] / (1 - self.mu ** self.t)
            new_param = param - self.lr * v_normed / (np.sqrt(g_normed) + self.eps)
            updated_params.append(new_param)
        return updated_params


class ParameterServer(object):

    def __init__(self, config):
        self.acc_grads_every_n = config.acc_grads_every_n
        self.db = RedisInterface(config.host, config.port)
        self.optimizer = Adam()

    def block_until_enough_n_of_grads(self):
        while self.db.get_n_of_grads_available() < self.acc_grads_every_n:
            sleep(0.5)

    def get_and_merge_grads(self):
        grads = self.db.get_n_first_grads(self.acc_grads_every_n)
        return [np.mean(i) for i in zip(*grads)]

    def apply_grads(self, params, grads):
        return self.optimizer.apply_grads(params, grads)
        # return [p - 0.001 * g for p, g in zip(params, grads)]

    def run(self):
        while True:
            print("Inside run")
            self.block_until_enough_n_of_grads()
            grads = self.get_and_merge_grads()
            params = self.db.get_params()
            updated_params = self.apply_grads(params, grads)
            self.db.set_params(updated_params)
            self.db.clear_grads_list()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Parameter server parameters',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--acc-every', type=int, dest="acc_grads_every_n", default=10)
    parser.add_argument('--host', type=str, help='Redis host', default="0.0.0.0")
    parser.add_argument('--port', dest='port', type=int, default=7070,
                        help='Port on which redis is listening')
    config = parser.parse_args()
    ParameterServer(config).run()
