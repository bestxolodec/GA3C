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
        self.params_prefix = "network_params"  # this should be same for all instances
        self.gradients_prefix = "gradients"  # this should be same for all instances
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

    def append_gradients(self, grads):
        self.connection.rpush(self.gradients_prefix, self._dump_obj_to_str(grads))

    def save_session(self, states, actions, rewards, memories=None):
        pass

    def load_params(self):
        return self._load_obj_from_str(self.connection.get(self.params_prefix))

    # should we have this one?
    def save_params(self, params):
        pass





