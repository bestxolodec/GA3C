
import numpy as np

from NetworkVP import NetworkVP
from Config import Config


# TODO: finish common interface
class AgentInterface(object):

    def get_actions(self, states):
        pass

    def get_gradients(self, states, actions, rewards, *args):
        pass

    def set_param_values(self, param_values):
        pass


class A3CAgent(object):

    # TODO: make init more general, this is ad_hoc way
    def __init__(self, n_actions):
        self.model = NetworkVP(Config.DEVICE, Config.NETWORK_NAME, n_actions)

    def get_gradients(self, states, actions, rewards, *args):
        return self.model.get_gradients(states, actions, rewards, *args)

    def set_param_values(self, param_values):
        self.model.set_all_trainable_param_values(param_values)

    def get_param_values(self):
        return self.model.get_all_trainable_param_values()

    # TODO remove this in favour of self.get_actions
    def predict_p_and_v(self, states):
        return self.model.predict_p_and_v(states)

    # TODO: this function is not currently in use (see value dependence problem -> TODO in ProcessAgent.py)
    def get_actions(self, states):
        probs, values = self.model.predict_p_and_v(states)
        if Config.PLAY_MODE:
            actions = np.argmax(probs, axis=1)
        else:
            actions = np.random.choice(self.actions, p=probs)
        return actions

