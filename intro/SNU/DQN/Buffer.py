"""
Data structure for implementing experience replay
"""
import random
import numpy as np


class Buffer(object):

    def __init__(self, buffer_size, terminal_buffer_size, state_dim, action_dim, reward_dim, random_seed=123):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.terminal_buffer_size = terminal_buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.count = 0
        self.terminal_count = 0
        self.test_count = 0
        self.database = np.array(([[]]))
        self.database_ter = np.array(([[]]))
        self.buffer = np.array(([[]]))
        self.terminal_buffer = np.array(([[]]))
#        self.test_buffer = np.array(([[]]))
        random.seed(random_seed)

    def data_add(self, args):
        experience = args
        if len(self.database[0]) == 0:
            self.database = experience
        else:
            self.database = np.concatenate((self.database, experience))

    def data_ter_add(self, args):
        experience = args
        if len(self.database_ter[0]) == 0:
            self.database_ter = np.array([experience])
        else:
            self.database_ter = np.concatenate((self.database_ter, np.array([experience])))

    def add(self, args):
        experience = args
        if self.count < self.buffer_size:
            if len(self.buffer[0]) == 0:
                self.buffer = experience
            else:
                self.buffer = np.concatenate((self.buffer, experience))

            self.count += 1

        else:
            self.buffer = np.delete(self.buffer, 0, 0)
            self.buffer = np.concatenate((self.buffer, experience))

    def terminal_add(self, args):
        experience = args
        if self.terminal_count < self.terminal_buffer_size:
            if len(self.terminal_buffer[0]) == 0:
                self.terminal_buffer = np.array([experience])
            else:
                self.terminal_buffer = np.concatenate((self.terminal_buffer, np.array([experience])))

            self.terminal_count += 1
        else:
            self.terminal_buffer = np.delete(self.terminal_buffer, 0, 0)
            self.terminal_buffer = np.concatenate((self.termianl_buffer, [experience]))

    def size(self):
        return self.count

    def terminal_size(self):
        return self.terminal_count

    def Make_buffer(self, number):
        if len(self.database) < number:
            sampled = self.database
        else:
            sampled = np.array(random.sample(list(self.database), number))
        self.add(sampled)

    def Make_buffer_ter(self, number):
        sampled = np.array(random.sample(list(self.database_ter), number))
        self.terminal_add(sampled)

    def sample_batch(self, batch_size):
        batch = []
        self.count = len(self.buffer)
        if self.count < batch_size:
            batch = random.sample(list(self.buffer), self.count)
        else:
            batch = random.sample(list(self.buffer), batch_size)

        s_batch = np.array([i[0:self.state_dim] for i in batch])
        a_batch = np.array([i[self.state_dim:self.state_dim+self.action_dim] for i in batch])
        r_batch = np.array([i[self.state_dim+self.action_dim:self.state_dim+self.action_dim+self.reward_dim] for i in batch])
        ss_batch = np.array([i[self.state_dim+self.action_dim+self.reward_dim:2*self.state_dim+self.action_dim+self.reward_dim] for i in batch])
        return s_batch, a_batch, r_batch , ss_batch

    def sample_terminal_batch(self, terminal_batch_size):
        if self.terminal_count < terminal_batch_size:
            terminal_batch = random.sample(list(self.terminal_buffer), self.terminal_count)
        else:
            terminal_batch = random.sample(list(self.terminal_buffer), terminal_batch_size)

        st_batch = np.array([i[0:self.state_dim] for i in terminal_batch])
        at_batch = np.array([i[self.state_dim:self.state_dim+self.action_dim] for i in terminal_batch])
        rt_batch = np.array([i[self.state_dim+self.action_dim:self.state_dim+self.action_dim+self.reward_dim] for i in terminal_batch])
        sst_batch = np.array([i[self.state_dim+self.action_dim+self.reward_dim:2*self.state_dim+self.action_dim+self.reward_dim] for i in terminal_batch])
        return st_batch, at_batch, rt_batch, sst_batch
