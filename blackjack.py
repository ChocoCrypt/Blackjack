import gym



class Agente_BlackJack:
    def __init__(self):
        self.env = gym.make("Blackjack-v1")
        print(self.env.observation_space)



if __name__=="__main__":
    agent = Agente_BlackJack()
    

