import gymnasium as gym
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium import spaces


CARDS_PER_AGENT = 7
NUM_AGENTS = 5
NUM_EPISODES = 1000
NUM_CARDS = 52
NUM_SUITS = 4
NUM_VALUE = 13

good = 0
bad = 0
draw = 0
reset_too_much = 0
total_reward = 0
ppo_episode_rewards = []
a2c_episode_rewards = []


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
        info = {}

        return observation, info

    def step(self, action):
        agent = self.agents[self.current_agent]
        global total_reward
        print(f"___________________________________________\nTura agenta {self.current_agent}")

        # print(f"\nRęka agenta przed turą:")
        # for card in agent.hand:
        #     print(f"{card}")
        # print("")

        if len(agent.hand) > 10:
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
            if action in (0, 1, 13, 14, 26, 27, 39, 40):#+2/+3
                reward+=100
            elif action in (11, 37):#+5
                reward+=200
            elif action in (2, 15, 28, 41):#skip
                reward += 50
            elif action in (12, 25, 38, 51):#AS
                reward +=25
            elif action in (9, 22, 35, 48):#WALET
                reward+=150

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

        # print(f"\nRęka agenta po turze:")
        # for card in agent.hand:
        #     print(f"{card}")
        # print(f"\nTop_card: {self.discard_pile.top_card()}")

        done = len(agent.hand) == 0

        if done:
            reward += 400

        observation = self.get_observation(self.current_agent)
        terminated = done
        truncated = False
        info = {}

        self.current_agent = (self.current_agent + 1) % NUM_AGENTS

        total_reward += reward

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


def make_env():
    return MacauEnv()


if __name__ == "__main__":
    # Ścieżki do katalogów z modelami
    ppo_models_dir = "models/PPO"
    a2c_models_dir = "models/A2C"

    # Tworzenie katalogów, jeśli nie istnieją
    os.makedirs(ppo_models_dir, exist_ok=True)
    os.makedirs(a2c_models_dir, exist_ok=True)

    # Ścieżki do najlepszych modeli
    ppo_model_path = os.path.join(ppo_models_dir, 'ppo_model')
    a2c_model_path = os.path.join(a2c_models_dir, 'a2c_model')

    # PPO
    if os.path.exists(ppo_model_path + ".zip"):
        print("Loading existing PPO model...")
        ppo_model = PPO.load(ppo_model_path)
        ppo_env = DummyVecEnv([make_env])
        ppo_model.set_env(ppo_env)
    else:
        print("Creating new PPO model...")
        ppo_env = DummyVecEnv([make_env])
        ppo_model = PPO('MlpPolicy', ppo_env, verbose=1)

    # Callback do ewaluacji i zapisywania najlepszego modelu PPO
    eval_callback_ppo = EvalCallback(ppo_env, best_model_save_path=ppo_models_dir,
                                     log_path=ppo_models_dir, eval_freq=10000,
                                     deterministic=True, render=False)

    # A2C
    if os.path.exists(a2c_model_path + ".zip"):
        print("Loading existing A2C model...")
        a2c_model = A2C.load(a2c_model_path)
        a2c_env = DummyVecEnv([make_env])
        a2c_model.set_env(a2c_env)
    else:
        print("Creating new A2C model...")
        a2c_env = DummyVecEnv([make_env])
        a2c_model = A2C('MlpPolicy', a2c_env, verbose=1)

    # Callback do ewaluacji i zapisywania najlepszego modelu A2C
    eval_callback_a2c = EvalCallback(a2c_env, best_model_save_path=a2c_models_dir,
                                     log_path=a2c_models_dir, eval_freq=10000,
                                     deterministic=True, render=False)

    # Trenowanie modelu PPO
    TIMESTEPS = 1  # Ustaw właściwą liczbę kroków
    ppo_episode_rewards = []
    for i in range(20):
        ppo_model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
        ppo_model.save(ppo_model_path)

        ppo_episode_rewards.append(total_reward)
        print(f"Agent PPO podejmuje złą akcje {bad} razy")
        print(f"Agent PPO podejmuje dobrą akcje {good} razy")
        print(f"Agent PPO zresetował środowisko z powodu zbyt wielu kart na ręce {reset_too_much} razy")
        total_reward = 0  # Przykład, dostosuj do swoich potrzeb
        bad = 0
        good = 0
        reset_too_much = 0

    # Trenowanie modelu A2C
    TIMESTEPS = 2048 # Ustaw właściwą liczbę kroków
    a2c_episode_rewards = []
    for i in range(20):
        a2c_model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
        a2c_model.save(a2c_model_path)

        a2c_episode_rewards.append(total_reward)
        print(f"Agent A2C podejmuje złą akcje {bad} razy")
        print(f"Agent A2C podejmuje dobrą akcje {good} razy")
        print(f"Agent A2C zresetował środowisko z powodu zbyt wielu kart na ręce {reset_too_much} razy")
        total_reward = 0  # Przykład, dostosuj do swoich potrzeb
        bad = 0
        good = 0
        reset_too_much = 0

    # Wykres nagród
    plt.plot(ppo_episode_rewards, label='PPO')
    plt.plot(a2c_episode_rewards, label='A2C')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.grid(True)
    plt.legend(['PPO', 'A2C'])
    plt.xticks(range(20))
    plt.show()

    #TODO wykresy:
    # 1. ten co jest
    # 2. zmiany średniej nagrody w czasie trwania pętli
    # 3. bad i good
    # 4. funkcja straty i jej pokazanie
    # 5. intrefejs prosty

