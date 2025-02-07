import time

import gymnasium as gym
import numpy as np
import random
import os

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium import spaces


CARDS_PER_AGENT = 6
NUM_AGENTS = 2
NUM_EPISODES = 1000
NUM_CARDS = 52
NUM_SUITS = 4
NUM_VALUE = 13

good = 0
bad = 0
draw = 0
reset_too_much = 0


class Card:
    def __init__(self, index, value, suit):
        self.index = index
        self.value = value
        self.suit = suit

    def __str__(self):
        return f"{self.index}. {self.value} {self.suit}"


class Deck:
    def __init__(self, macau_env):
        self.cards = self._create_deck()
        self.macau_env = macau_env

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

    def realize_effects(self, card, reward):
        if card.value == 2:
            self.draw(2, reward)
        elif card.value == 3:
            self.draw(3, reward)
        elif card.value == 4:
            self.lose_turn(reward)
        elif card.value == 11:
            self.value_request(reward)
        elif card.value == 13 and card.suit in [1, 3]:
            self.draw(5, reward)
        elif card.value == 14:
            self.change_suite(reward)
        return reward

    def draw(self, value, reward):
        for _ in range(value):
            self.macau_env.agents[(self.macau_env.current_agent + 1) % NUM_AGENTS].hand.append(self.macau_env.deck.draw_card(self.macau_env.discard_pile))
        reward += 5 * value
        return reward

    def lose_turn(self, reward):
        self.macau_env.current_agent = (self.macau_env.current_agent + 1) % NUM_AGENTS
        reward += 5
        return reward

    def value_request(self, reward):
        current_agent_hand = self.macau_env.agents[self.macau_env.current_agent].hand

        # Stwórz słownik wartości kart do ich indeksów dla aktualnego agenta
        value_to_indices = {}
        for card in current_agent_hand:
            if card.value not in value_to_indices:
                value_to_indices[card.value] = []
            value_to_indices[card.value].append(card.index)

        # Znajdź wartość karty, która najczęściej występuje w ręce aktualnego agenta
        if value_to_indices:
            requested_value = max(value_to_indices, key=lambda k: len(value_to_indices[k]))

            # Przejdź przez ręce pozostałych agentów i sprawdź, czy mają kartę o tej wartości
            for i in range(1, NUM_AGENTS):
                current_agent_id = (self.macau_env.current_agent + i) % NUM_AGENTS
                agent_hand = self.macau_env.agents[current_agent_id].hand

                matching_cards = [card for card in agent_hand if card.value == requested_value]

                if matching_cards:
                    card_to_play = matching_cards[0]
                    self.macau_env.discard_pile.discard(card_to_play)
                    agent_hand.remove(card_to_play)
                    print(f"Agent {current_agent_id} zagrał kartę {card_to_play}")
                else:
                    drawn_card = self.macau_env.deck.draw_card(self.macau_env.discard_pile)
                    agent_hand.append(drawn_card)
                    print(f"Agent {current_agent_id} dobrał kartę {drawn_card}")
                    reward += 2

        return reward

    def change_suite(self, reward):
        # Pobierz rękę aktualnego agenta
        current_agent_hand = self.macau_env.agents[self.macau_env.current_agent].hand

        # Stwórz słownik kolorów kart do ich indeksów dla aktualnego agenta
        suit_to_indices = {}
        for card in current_agent_hand:
            if card.suit not in suit_to_indices:
                suit_to_indices[card.suit] = []
            suit_to_indices[card.suit].append(card.index)

        # Znajdź kolor karty, który najczęściej występuje w ręce aktualnego agenta
        if suit_to_indices:
            self.macau_env.requested_suit = max(suit_to_indices, key=lambda k: len(suit_to_indices[k]))

        agent_hand = self.macau_env.agents[(self.macau_env.current_agent + 1) % NUM_AGENTS].hand
        card = next((c for c in agent_hand if c.suit == self.macau_env.requested_suit), None)
        reward += 5 if card is None else 0

        return reward


class Agent:
    def __init__(self, agent_id, observation_space, action_space):
        self.agent_id = agent_id
        self.observation_space = observation_space
        self.action_space = action_space
        self.hand = []


def encode_hand(agent_hand):
    encoded = np.zeros(NUM_CARDS, dtype=np.float32)
    for card in agent_hand:
        encoded[card.index] = 1
    return encoded


def encode_top_card(top_card):
    encoded = np.zeros(NUM_CARDS, dtype=np.float32)
    encoded[top_card.index] = 1
    return encoded


def encode_top_card_suit_value(top_card):
    suit_encoded = np.zeros(NUM_SUITS, dtype=np.float32)
    value_encoded = np.zeros(NUM_VALUE, dtype=np.float32)
    suit_encoded[top_card.suit] = 1
    value_encoded[top_card.value - 2] = 1
    return suit_encoded, value_encoded


class MacauEnv(gym.Env):
    def __init__(self):
        super(MacauEnv, self).__init__()
        self.deck = Deck(self)
        self.discard_pile = DiscardPile()
        self.agents = [Agent(agent_id, None, None) for agent_id in range(NUM_AGENTS)]
        self.current_agent = 0
        self.requested_suit = None

        self.action_space = spaces.Discrete(NUM_CARDS + 1)  # 0-51 rzut karty, 52 dobranie karty
        self.observation_space = spaces.Box(low=0, high=1, shape=(NUM_CARDS + NUM_CARDS + NUM_SUITS + NUM_VALUE,), dtype=np.float32)

    def reset(self, seed=None, **kwargs):
        self.deck = Deck(self)
        self.deck.shuffle()
        self.discard_pile = DiscardPile()
        self.agents = [Agent(agent_id, self.observation_space, self.action_space) for agent_id in range(NUM_AGENTS)]
        self.current_agent = 0
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

        print(f"___________________________________________\nTura agenta {self.current_agent}")

        print(f"\nRęka agenta przed turą:")
        for card in agent.hand:
            print(f"{card}")
        print("")
        if len(agent.hand) > 20:
            global reset_too_much
            reset_too_much += 1
            reward = -10000
            observation = self.get_observation(self.current_agent)
            terminated = True
            truncated = False
            info = {}
            return observation, reward, terminated, truncated, info


        if self.is_action_correct(action):  # legalna akcja
            global good
            good += 1
            reward = 100
            if action == 52:
                drawn_card = self.deck.draw_card(self.discard_pile)
                agent.hand.append(drawn_card)
                print(f"Agent {self.current_agent} dobral karte {drawn_card}")
                reward += -2000 if len(self.valid_action_space(self.current_agent)) > 1 else 0
            else:
                card = next(c for c in agent.hand if c.index == action)
                self.discard_pile.discard(card)
                agent.hand.remove(card)
                print(f"Agent {self.current_agent} zagral karte {card}")

                reward += CardsEffects(self).realize_effects(card, reward)
        else:                               # nielegalna akcja
            global bad
            bad += 1
            reward = -100
            observation = self.get_observation(self.current_agent)
            terminated = False
            truncated = False
            info = {'mask': self.get_action_mask(self.current_agent)}
            return observation, reward, terminated, truncated, info

        print(f"\nRęka agenta po turze:")
        for card in agent.hand:
            print(f"{card}")
        print(f"\nTop_card: {self.discard_pile.top_card()}")

        done = len(agent.hand) == 0

        if done:
            reward += 400

        observation = self.get_observation(self.current_agent)
        terminated = done
        truncated = False
        info = {'mask': self.get_action_mask(self.current_agent)}

        self.current_agent = (self.current_agent + 1) % NUM_AGENTS

        return observation, reward, terminated, truncated, info

    def get_observation(self, agent_index):
        agent_hand = self.agents[agent_index].hand
        top_card = self.discard_pile.top_card()

        hand_encoded = encode_hand(agent_hand)
        top_card_encoded = encode_top_card(top_card)
        suit_encoded, value_encoded = encode_top_card_suit_value(top_card)
        observation = np.concatenate([hand_encoded, top_card_encoded, suit_encoded, value_encoded])
        return observation

    def valid_action_space(self, agent_index):
        agent_hand = self.agents[agent_index].hand
        top_card = self.discard_pile.top_card()
        valid_actions = []

        for card in agent_hand:
            if self.requested_suit is not None:
                if card.suit == self.requested_suit or card.value == top_card.value or card.value == 12 or top_card.value == 12:
                    valid_actions.append(card.index)
                self.requested_suit = None

            else:
                if card.value == top_card.value or card.suit == top_card.suit or top_card.value == 12 or card.value == 12:
                    valid_actions.append(card.index)

        valid_actions.append(52)  # Dobranie karty
        return valid_actions

    def get_action_mask(self, agent_index):
        valid_actions = self.valid_action_space(agent_index)
        mask = np.zeros(NUM_CARDS + 1, dtype=np.float32)
        for action in valid_actions:
            mask[action] = 1
        return mask

    def is_action_correct(self, action):
        valid_actions = self.valid_action_space(self.current_agent)  # Replace agent_index with the appropriate index
        correct = action in valid_actions
        return correct


if __name__ == "__main__":
    models_dir = f"models/PPO"
    logdir = f"logs/PPO"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    env = MacauEnv()
    check_env(env)
    vec_env = DummyVecEnv([lambda: env])
    model = PPO('MlpPolicy', vec_env, verbose=1, tensorboard_log=logdir)

    TIMESTEPS = 1000
    for i in range(1, 200):
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
        model.save(f"{models_dir}/{TIMESTEPS * i}")

    print(f"\nAgent wybrał dobrą akcję {good} razy")
    print(f"\nAgent wybrał złą akcję {bad} razy")
