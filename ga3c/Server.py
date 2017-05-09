# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from multiprocessing import Queue

import time
import numpy as np

from Config import Config
from Environment import Environment
from ProcessAgent import ProcessAgent
from ProcessStats import ProcessStats
from ThreadDynamicAdjustment import ThreadDynamicAdjustment
from ThreadPredictor import ThreadPredictor
from ThreadTrainer import ThreadTrainer
from Database import RedisInterface
from Agent import A3CAgent

class Server:
    def __init__(self):
        self.db = RedisInterface(Config.DB_HOST, Config.DB_PORT)
        self.stats = ProcessStats()

        self.training_q = Queue(maxsize=Config.MAX_QUEUE_SIZE)
        self.prediction_q = Queue(maxsize=Config.MAX_QUEUE_SIZE)

        self.agent = A3CAgent(Environment().get_num_actions())
        if self.db.get_params() is None:  # initial params in db are not set yet
            self.db.set_params(self.agent.get_param_values())

        self.agents = []
        self.predictors = []
        self.trainers = []
        self.dynamic_adjustment = ThreadDynamicAdjustment(self)

    def add_agent(self):
        self.agents.append(
            ProcessAgent(len(self.agents), self.prediction_q, self.training_q, self.stats.episode_log_q))
        self.agents[-1].start()

    def add_predictor(self):
        self.predictors.append(ThreadPredictor(self, len(self.predictors)))
        self.predictors[-1].start()

    def add_trainer(self):
        self.trainers.append(ThreadTrainer(self, len(self.trainers)))
        self.trainers[-1].start()

    @staticmethod
    def remove_from(entities_list):
        entities_list[-1].exit_flag = True
        entities_list[-1].join()
        entities_list.pop()

    def remove_agent(self): self.remove_from(self.agents)

    def remove_predictor(self): self.remove_from(self.predictors)

    def remove_trainer(self): self.remove_from(self.trainers)

    def train_model(self, x_, a_, r_, trainer_id):
        grads = self.agent.get_gradients(x_, a_, r_, trainer_id)
        self.db.append_gradients(grads)
        self.stats.training_count.value += 1
        self.dynamic_adjustment.temporal_training_count += 1
        self.db.add_session(x_, a_, r_)

    def main(self):
        self.stats.start()
        self.dynamic_adjustment.start()

        if Config.PLAY_MODE:
            for trainer in self.trainers:
                trainer.enabled = False

        params_modify_time = self.db.get_params_modify_time()
        while self.stats.episode_count.value < Config.EPISODES:
            if params_modify_time < self.db.get_params_modify_time():
                self.agent.set_param_values(self.db.get_params())
                # from datetime import datetime as dt
                # latency = dt.now() - self.db.get_params_modify_time()
                # params_raveled = tuple(np.hstack([i.ravel() for i in self.agent.get_param_values()]))
                # logging.debug("Latency: {}, current weights hash: {}".format(latency, hash(params_raveled)))
                params_modify_time = self.db.get_params_modify_time()

            # jiter 0..0.5s to prevent network congestion
            time.sleep(np.random.rand() / 2)

        self.dynamic_adjustment.exit_flag = True
        while self.agents:
            self.remove_agent()
        while self.predictors:
            self.remove_predictor()
        while self.trainers:
            self.remove_trainer()
