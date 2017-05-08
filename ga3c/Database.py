import os
import socket
import redis
import joblib
import logging
from io import BytesIO


class Database(object):
    """Abstract class, TBD"""
    pass


class RedisDB(Database):

    def __init__(self, host, port):
        self.host = host  # TODO: do we need to store host and port?
        self.port = port
        self.prefix = "{}.{}.".format(socket.gethostname(),  os.getpid())

        try:
            self.connection = redis.Redis(host=self.host, port=self.port)
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



    def save_session(self, observations, actions, rewards, is_alive ):



    def save_params(self, params):
        pass

    def load_params(self, params):
        pass






