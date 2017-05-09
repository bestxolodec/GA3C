import os
import socket
import redis
import joblib
import logging
from io import BytesIO


class DatabaseInterface(object):
    """Abstract class, TBD"""
    pass


class RedisInterface(DatabaseInterface):

    def __init__(self, host, port):
        self.host = host  # TODO: do we need to store host and port?
        self.port = port
        self.prefix = "{}.{}.".format(socket.gethostname(),  os.getpid())
        self.params_key = "params"  # this should be same for all instances
        self.gradients_key = "gradients"  # this should be same for all instances
        self.connection = redis.Redis(host=self.host, port=self.port)
        try:
            self.connection.client_list()
        except redis.ConnectionError:
            logging.FATAL("Redis on {}:{} could not be reached!".format(self.host, self.port))
            raise redis.ConnectionError

    @staticmethod
    def _dump_obj_to_str(obj):
        binary_stream = BytesIO()
        joblib.dump(obj, binary_stream)
        return binary_stream.getvalue()

    @staticmethod
    def _load_obj_from_str(string):
        return joblib.load(BytesIO(string))

    def get_params(self):
        # TODO need to implement logic with checking time modification
        # http://stackoverflow.com/a/9917360/2046408
        params_str = self.connection.get(self.params_key)
        return None if params_str is None else self._load_obj_from_str(params_str)

    def set_params(self, params):
        self.connection.set(self.params_key, self._dump_obj_to_str(params))

    def append_gradients(self, grads):
        self.connection.rpush(self.gradients_key, self._dump_obj_to_str(grads))

    def get_n_of_grads_available(self):
        return self.connection.llen(self.gradients_key)

    def get_n_first_grads(self, n_first):
        list_of_grads = self.connection.lrange(self.gradients_key, 0, n_first)
        return [self._load_obj_from_str(s) for s in list_of_grads]

    def clear_grads_list(self):
        self.connection.delete(self.gradients_key)

    def save_session(self, states, actions, rewards, memories=None):
        raise NotImplementedError




