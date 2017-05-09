import os
import socket
from datetime import datetime

import redis
import joblib
import logging
from io import BytesIO


# TODO: fill in this abstrac interface
class DatabaseInterface(object):
    """Abstract class, TBD"""
    pass


class RedisInterface(DatabaseInterface):

    def __init__(self, host, port):
        self.host = host  # TODO: do we need to store host and port?
        self.port = port
        self.prefix = "{}.{}.".format(socket.gethostname(),  os.getpid())
        self.params_key = "params"  # this should be same for all instances
        self.params_modifed_key = "params.timestamp"  # this should be same for all instances
        self.gradients_key = "gradients"  # this should be same for all instances
        self.sessions_key = "sessions"  # this should be same for all instances
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
        params_str = self.connection.get(self.params_key)
        return None if params_str is None else self._load_obj_from_str(params_str)

    def set_params(self, params):
        self.connection.set(self.params_key, self._dump_obj_to_str(params))
        self.connection.set(self.params_modifed_key,  self._dump_obj_to_str(datetime.now()))

    def get_params_modify_time(self):
        return self._load_obj_from_str(self.connection.get(self.params_modifed_key))

    def append_gradients(self, grads):
        # TODO: what if parameter server has been shutted down? we will get infinte size list - FIX needed
        self.connection.rpush(self.gradients_key, self._dump_obj_to_str(grads))

    def get_n_of_grads_available(self):
        return self.connection.llen(self.gradients_key)

    def get_n_first_grads(self, n_first):
        list_of_grads = self.connection.lrange(self.gradients_key, 0, n_first)
        return [self._load_obj_from_str(s) for s in list_of_grads]

    # This function probably suffice and previos should be deleted
    def get_all_grads(self):
        return self.get_n_first_grads(-1)

    def clear_grads_list(self):
        self.connection.delete(self.gradients_key)

    # TODO: BELOW ARE TWO UNTESTED FUNCTIONS, BEWARE
    # FIXME: experimental feature
    def add_session(self, states, actions, rewards, initial_memory=None):
        data = [states, actions, rewards, initial_memory]  # not sure if this format is optimal
        self.connection.sadd(self.sessions_key, self._dump_obj_to_str(data))

    # FIXME: experimental function
    def get_sessions(self, n_sessions):
        sesisons = self.connection.srandmember(self.sessions_key, n_sessions)
        return [self._load_obj_from_str(s) for s in sesisons]



