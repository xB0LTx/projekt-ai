import gymnasium as gym
import numpy as np
import random
import os

from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium import spaces




CARDS_PER_AGENT = 6
NUM_AGENTS = 4
NUM_EPISODES = 1000
good = 0
bad = 0
draw = 0

class Card:
    def __init__(self, index, value, suit):
        self.index = index
        self.value = value
        self.suit = suit

    def __str__(self):
        return f"{self.index}. {self.value} {self.suit}"


class Deck:
    def __init__(self):
        self.cards = self._create_deck()

    def _create_deck(self):
        values = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        suits = [0, 1, 2, 3]
        deck = []
        for i, suit in enumerate(suits):
            for j, value in enumerate(values):
                index = i * len(values) + j
                deck.append(Card(index, value, suit))
        return deck

    def shuffle(self):
        random.shuffle(self.cards)

    def draw_card(self, discard_pile):
        if len(self.cards) == 0:
            self.cards = discard_pile.discard_pile_list[:-1]
            discard_pile.discard_pile_list = discard_pile.discard_pile_list[-1:]
            self.shuffle()
        return self.cards.pop()

    def deal_cards(self, agents, discard_pile):
        for agent in agents:
            agent.hand = []

        for _ in range(CARDS_PER_AGENT):
            for agent_id in range(NUM_AGENTS):
                card = self.draw_card(discard_pile)
                agents[agent_id].hand.append(card)


class DiscardPile:
    def __init__(self):
        self.discard_pile_list = []

    def discard(self, discarded_card):
        self.discard_pile_list.append(discarded_card)

    def top_card(self):
        return self.discard_pile_list[-1]


class CardsEffects:
    def __init__(self, macau_env):
        self.macau_env = macau_env

    def realize_effects(self, card):
        if card.value == 2:
            self.draw(2)
        elif card.value == 3:
            self.draw(3)
        elif card.value == 4:
            self.lose_turn()
        elif card.value == 11:
            self.value_request()
        elif card.value == 12:
            self.make_uniwersal_card()
        elif card.value == 13 and card.suit in [1, 3]:
            self.draw(5)
        elif card.value == 14:
            self.change_suite()

    def draw(self, value):
        for _ in range(value):
            self.macau_env.agents[(self.macau_env.current_agent + 1) % NUM_AGENTS].hand.append(self.macau_env.deck.draw_card(self.macau_env.discard_pile))

    def lose_turn(self):
        self.macau_env.current_agent = (self.macau_env.current_agent + 1) % NUM_AGENTS

    def value_request(self):
        requested_value = self.macau_env.requested_value

        # value_to_indices = {}
       #  for card in agent_hand:
       #      if card.value not in value_to_indices:
       #          value_to_indices[card.value] = []
       #      value_to_indices[card.value].append(card.index)
       #
       #  requested_value = max(value_to_indices, key=lambda k: len(value_to_indices[k]))

        for i in range(1, NUM_AGENTS):
            current_agent_id = (self.macau_env.current_agent + i) % NUM_AGENTS
            agent_hand = self.macau_env.agents[current_agent_id].hand
            matching_cards = [card for card in agent_hand if card.value == requested_value]

            if matching_cards:
                card_to_play = matching_cards[0]
                self.macau_env.discard_pile.discard(card_to_play)
                agent_hand.remove(card_to_play)
                print(f"Agent {current_agent_id} zagral karte {card_to_play}")
            else:
                drawn_card = self.macau_env.deck.draw_card(self.macau_env.discard_pile)
                agent_hand.append(drawn_card)
                print(f"Agent {current_agent_id} dobral karte {drawn_card}")

    def make_uniwersal_card(self):
        pass

    def change_suite(self):
        pass


class Agent:
    def __init__(self, agent_id, observation_space, action_space):
        self.agent_id = agent_id
        self.observation_space = observation_space
        self.action_space = action_space
        self.hand = []


class MacauEnv(gym.Env):
    def __init__(self):
        super(MacauEnv, self).__init__()
        self.deck = Deck()
        self.discard_pile = DiscardPile()
        self.agents = [Agent(agent_id, None, None) for agent_id in range(NUM_AGENTS)]
        self.current_agent = 0
        self.requested_value = None
        self.requested_suit = None

        self.action_space = spaces.MultiDiscrete([53, 7, 5])  # [(0-52), (0-5), (0-3)]
        self.observation_space = spaces.Box(low=-1, high=52, shape=(53 * 3,), dtype=np.int32)

    def reset(self, seed=None):
        self.deck = Deck()
        self.deck.shuffle()
        self.discard_pile = DiscardPile()
        self.agents = [Agent(agent_id, self.observation_space, self.action_space) for agent_id in range(NUM_AGENTS)]
        self.current_agent = 0
        self.requested_value = None
        self.requested_suit = None

        self.deck.deal_cards(self.agents, self.discard_pile)

        initial_card = self.deck.draw_card(self.discard_pile)
        self.discard_pile.discard(initial_card)

        print(f"\nKarta poczatkowa: {initial_card}")
        observation = self.get_observation(self.current_agent)
        info = {'mask': self.get_action_mask(self.current_agent)}

        return observation, info

    def step(self, action):
        agent = self.agents[self.current_agent]

        action_type, requested_value, requested_suit = action

        correct = self.is_action_correct(action, agent)

        if correct:
            global good
            good += 1
            reward = 500000
        else:
            global bad
            bad += 1
            observation = self.get_observation(self.current_agent)
            reward = -100000
            terminated = False
            truncated = False
            info = {}
            return observation, reward, terminated, truncated, info


        print(f"___________________________________________\nTura agenta {self.current_agent}")

        print(f"\nRęka agenta przed turą:")
        for card in agent.hand:
            print(f"{card}")
        print("")

        if action_type == 52:
            global draw
            draw =+ 1
            drawn_card = self.deck.draw_card(self.discard_pile)
            agent.hand.append(drawn_card)
            print(f"Agent {self.current_agent} dobral karte {drawn_card}")
        else:
            card = next(c for c in agent.hand if c.index == action_type)
            self.discard_pile.discard(card)
            agent.hand.remove(card)
            print(f"Agent {self.current_agent} zagral karte {card}")

            if card.value == 11:
                self.requested_value = requested_value
            elif card.value == 14:
                self.requested_suit = requested_suit

            CardsEffects(self).realize_effects(card)

        print(f"\nRęka agenta po turze:")
        for card in agent.hand:
            print(f"{card}")
        print(f"\nTop_card: {self.discard_pile.top_card()}")

        done = len(agent.hand) == 0

        if done:
            reward += 50

        observation = self.get_observation(self.current_agent)
        terminated = done
        truncated = False
        info = {'mask': self.get_action_mask(self.current_agent)}

        self.current_agent = (self.current_agent + 1) % NUM_AGENTS

        return observation, reward, terminated, truncated, info

    def get_observation(self, agent_index):
        agent_hand = self.agents[agent_index].hand
        top_card = self.discard_pile.top_card()

        # Utwórz tablicę numpy o rozmiarze 53x3
        observation = np.zeros((53, 3), dtype=np.int32)

        # Wypełnij pierwsze 52 wiersze danymi z ręki agenta
        for i, card in enumerate(agent_hand):
            observation[i] = [card.index, card.value, card.suit]

        # Wypełnij ostatnie 3 wiersze danymi z karty na wierzchu stosu odrzutów
        if top_card:
            observation[52] = [top_card.index, top_card.value, top_card.suit]

        #   TODO sprawdzić czy te obserwacje będę wystarczające (hand i top_card)
        return observation.flatten()

    def valid_action_space(self, agent_index):
        agent_hand = self.agents[agent_index].hand
        top_card = self.discard_pile.top_card()
        valid_actions = []

        for card in agent_hand:
            if card.value == top_card.value or card.suit == top_card.suit or top_card.value == 12 or card.value == 12:
                if card.value == 11:  # Karty wymagające wyboru jednego z 6 efektów
                    for value in range(6):
                        valid_actions.append([card.index, value + 1, 0])
                elif card.suit == 14:  # Karty wymagające wyboru jednego z 4 efektów
                    for suit in range(4):
                        valid_actions.append([card.index, 0, suit + 1])
                else:
                    valid_actions.append([card.index, 0, 0])
        valid_actions.append([52, 0, 0])  # Dobranie karty
        return valid_actions

    def is_action_correct(self, action, agent):
        action_type, requested_value, requested_suit = action
        valid_actions = self.valid_action_space(self.current_agent)

        is_valid_action = any(
            action[0] == valid_action[0] and
            action[1] == valid_action[1] and
            action[2] == valid_action[2]
            for valid_action in valid_actions
        )

        additional_conditions = (
            (action_type in [9, 22, 35, 48] and requested_value in [1, 2, 3, 4, 5, 6] and requested_suit == 0) or
            (action_type in [12, 25, 38, 51] and requested_value == 0 and requested_suit in [1, 2, 3, 4]) or
            (action_type in range(53) and action_type not in [9, 22, 35, 48, 12, 25, 38, 51] and requested_value == 0 and requested_suit == 0)
        )

        return is_valid_action and additional_conditions

    def get_action_mask(self, agent_index):
        valid_actions = self.valid_action_space(agent_index)
        mask = np.zeros(self.action_space.nvec, dtype=int)
        for action in valid_actions:
            mask[action[0], action[1], action[2]] = 1
        return mask


if __name__ == "__main__":
    models_dir = "models/A2C"
    logdir = "logs"

    if not os.path.exists((models_dir)):
        os.makedirs((models_dir))

    if not os.path.exists((logdir)):
        os.makedirs((logdir))

    env = MacauEnv()
    check_env(env)
    vec_env = DummyVecEnv([lambda: MacauEnv()])

    #model = PPO("MlpPolicy", vec_env, verbose=1)
    model = A2C("MlpPolicy", vec_env, verbose=1)

    TIMESTEPS = 1000
    for i in range (30):
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="A2C")
        model.save(f"{models_dir}/{TIMESTEPS*i}")

    print(f"\nAgent wybrał dobrą akcję {good} razy")
    print(f"\nAgent wybrał złą akcję {bad} razy")


