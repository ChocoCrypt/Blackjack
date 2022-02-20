import gym
from pprint import pprint
import random
from time import sleep
import numpy as np




class Agent_BlackJack:
    """
    Reinforcement Learning agent that is supposed to learn playing OpenAI's
    blackjack game by the implementation of Monte Carlo Control algorithm.
    """
    def __init__(self):
        self.env = gym.make("Blackjack-v1")
        #print(self.env.action_space)
        observation_space = self.env.observation_space
        action_space = self.env.action_space
        print(observation_space)
        print(action_space)
        states = [i for i in range(32)]
        cards = [i for i in range(11)]
        actions = [0,1]
        self.all_states = {}
        cont = 1
        for state in states:
            for card in cards:
                self.all_states[(state,card)] = cont
                cont +=1
        pprint(self.all_states)
        dim = (len(self.all_states.values()), len([0,1]) )
        self.Q = np.random.uniform(-2,0 , dim)
        print(self.Q)

        
    def search_in_q(self,state,card):
        val = self.all_states[(state,card)]
        print(val)
        return(val)
        

    def play(self):
        done = False
        new_state = self.env.reset()
        while(not done):
            self.env.render()
            act_in_table  = np.argmax(self.search_in_q(new_state[0] , new_state[1]))
            action = np.argmax(self.Q[act_in_table])
            new_state, reward, done, info = self.env.step(action)
            print(f"new state = {new_state}")
            print(f"reward = {reward}")
            print(f"done = {done}")
            print(f"info = {info}")
            sleep(1)

if __name__ == "__main__":
    agent = Agent_BlackJack()
    agent.play()

