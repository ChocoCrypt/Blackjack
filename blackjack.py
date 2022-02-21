import gym
from tqdm import tqdm
from pprint import pprint
import random
from time import sleep
import numpy as np




class Agent_BlackJack:
    """
    Reinforcement Learning agent that is supposed to learn playing OpenAI's
    blackjack game by the implementation of Monte Carlo Control algorithm.
    """
    def __init__(self, delta ,gamma , epsilon):
        self.LEARNING_RATE = delta
        self.DISCOUNT = gamma
        self.epsilon = epsilon
        self.env = gym.make("Blackjack-v1")
        #print(self.env.action_space)
        observation_space = self.env.observation_space
        action_space = self.env.action_space
        states = [i for i in range(32)]
        cards = [i for i in range(11)]
        actions = [0,1]
        self.all_states = {}
        cont = 1
        for state in states:
            for card in cards:
                self.all_states[(state,card)] = cont
                cont +=1
        dim = (len(self.all_states.values()) +1 , len([0,1]) )
        self.Q = np.random.uniform(-2,0 , dim)

        
    def search_in_q(self,state,card):
        val = self.all_states[(state,card)]
        return(val)
        
    def get_action(self, state,card):
        choice = np.random.uniform(0,1)
        if(choice > self.epsilon):
            act_in_table  = np.argmax(self.search_in_q(state, card))
            action =np.argmax(self.Q[act_in_table])
        else:
            action = random.choice([0,1])
            self.epsilon = self.epsilon*0.9
        #print(f"for {state} and {card} selected action {action}")
        return(action)

    def get_max_future_q(self,state,card):
        val = self.search_in_q(state,card)
        max_future = np.max(self.Q[val])
        return(max_future)

    def play(self):
        done = False
        new_state = self.env.reset()
        while(not done):
            #self.env.render()
            action = self.get_action(new_state[0] , new_state[1])
            new_state, reward, done, info = self.env.step(action)
            """
            Here i have to update Q matrix applying Q learning formula
            """
            max_future_q = self.get_max_future_q(new_state[0],new_state[1])
            #new_Q = (1 - self.LEARNING_RATE ) * self.Q + self.LEARNING_RATE * (reward + self.DISCOUNT + max_future_q )
            #print(new_Q)
            #self.Q[self.search_in_q(new_state[0] , new_state[1])] = new_Q
            val_in_table = self.search_in_q(new_state[0] , new_state[1])
            ant = self.Q[val_in_table]
            new_q_value = (1-self.LEARNING_RATE) * ant  +  self.LEARNING_RATE*(reward + self.DISCOUNT*max_future_q)
            self.Q[val_in_table] = new_q_value
            if(new_state[2]):
                return(True)
        return False
                #sleep(3)

    def train(self,n_epochs):
        proms = [0]
        for i in tqdm(range(n_epochs)):
            ganamos = self.play()
            if(ganamos):
                proms.append(1)
            else:
                proms.append(0)
            if(i%100_000 == 0):
                print(np.mean(proms))
                proms = [0]



if __name__ == "__main__":
    agent = Agent_BlackJack(1.0 , 0.0 , 0.0000001)
    agent.train(1_000_000)

