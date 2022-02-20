import gym
from gym  import spaces
import numpy as np



class Agent_BlackJack:
    """
    The idea beside training an agent to play blackjack is to learn by Monte
    Carlo Control
    """
    def __init__(self , epsilon , gamma):
        self.epsilon = epsilon
        self.gamma = gamma
        self.env = gym.make("Blackjack-v1")
        self.total_rewards = []
        self.action_space = spaces.Discrete(2)
        print(self.action_space)
        self.done = False
        self.state_space = []
        self.Q = {}
        self.returns = {}
        self.pairs_visited = {}
        agent_sum_space = [i for i in range(4,22)]
        dealer_show_card_space = [i+1 for i in range(10)]
        self.action_space = [0,1]
        action_space = [0,1] #repeated lol
        agentAceSpace = [True,False]
        for total in agent_sum_space:
            for card in dealer_show_card_space:
                for ace in agentAceSpace:
                    for action in action_space:
                        self.Q[((total,card,ace),action)] = 0
                        self.returns[((total,card,ace),action)] = 0
                        self.pairs_visited[((total,card,ace),action)] = 0
                    self.state_space.append((total,card,ace))
        print(self.Q)

        self.policy = {}
        for state in self.state_space:
            print(state)
            self.policy[state] = np.random.choice(self.action_space)
            print(np.random.choice(self.action_space))


    def train(self, n_epochs):
        for _ in range(n_epochs):
            states_action_returns = []
            memory = []
            

if __name__=="__main__":
    agent = Agent_BlackJack(0.05 , 1)
    agent.train(10)
    

