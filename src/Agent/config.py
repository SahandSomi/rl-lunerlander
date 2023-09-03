class DQN_soft_update():

    def __init__(self):

        self.BUFFER_SIZE = int(1e4)  # replay buffer size
        self.BATCH_SIZE = 64         # minibatch size
        self.GAMMA = 0.99            # discount factor
        self.TAU = 1e-3              # for soft update of target parameters
        self.LR = 5e-4              # learning rate 
        self.UPDATE_EVERY = 4        # how often to update the network
        self.SEED = 0