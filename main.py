from collections import deque
import random
import numpy as np
import gym
from gym import spaces
import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Flatten, Dense

CARDS_PER_AGENT = 6
NUM_AGENTS = 4
NUM_EPISODES = 1000


def create_q_model():
    model = Sequential()
    model.add(Flatten(input_shape=(54,)))  # 52 karty + top_card + requested_value
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(59, activation='linear'))  # 52 akcje kart + 1 dobieranie + 6 wartości (5-10)
    return model


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001 
        self.model = create_q_model()
        self.model.compile(optimizer=tf.python.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, valid_actions):
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)
        act_values = self.model.predict(state)
        masked_values = [act_values[0][i] if i in valid_actions else -np.inf for i in range(self.action_size)]
        return np.argmax(masked_values)

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

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
        values = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'As']
        suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
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
        if card.value == '2':
            self.draw(2)
        elif card.value == '3':
            self.draw(3)
        elif card.value == '4':
            self.lose_turn()
        elif card.value == 'J':
            self.value_request()
        elif card.value == 'Q':
            self.make_uniwersal_card()
        elif card.value == 'K' and card.suit in ['Spades', 'Hearts']:
            self.draw(5)
        elif card.value == 'As':
            self.change_suite()

    def draw(self, value):
        for _ in range(value):
            self.macau_env.agents[(self.macau_env.current_agent + 1) % NUM_AGENTS].hand.append(self.macau_env.deck.draw_card(self.macau_env.discard_pile))

    def lose_turn(self):
        self.macau_env.current_agent = (self.macau_env.current_agent + 1) % NUM_AGENTS

    def value_request(self):
        requested_value = self.macau_env.requested_value
        print(f"Agent {self.macau_env.current_agent} zada wartosci: {requested_value}")

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

        self.action_space = spaces.Discrete(59)  # 52 karty + 1 dobranie karty + 6 wartosci (5-10)
        self.observation_space = spaces.Box(low=0, high=1, shape=(54,), dtype=np.float32)

    def reset(self):
        self.deck = Deck()
        self.deck.shuffle()
        self.discard_pile = DiscardPile()
        self.agents = [Agent(agent_id, self.observation_space, self.action_space) for agent_id in range(NUM_AGENTS)]
        self.current_agent = 0
        self.requested_value = None

        self.deck.deal_cards(self.agents, self.discard_pile)

        initial_card = self.deck.draw_card(self.discard_pile)
        self.discard_pile.discard(initial_card)

        print(f"\nKarta poczatkowa: {initial_card}")

        return self.get_observation(self.current_agent)

    def step(self, action):
        agent = self.agents[self.current_agent]

        print(f"___________________________________________\nTura agenta {self.current_agent}")

        if isinstance(action, tuple):
            card_action, requested_value_idx = action
            self.requested_value = requested_value_idx - 48
            print(f"Agent {self.current_agent} zada wartosci: {self.requested_value}")
            self.make_action(card_action, agent)
        elif isinstance(action, list):
            for idx in action:
                card = next(c for c in agent.hand if c.index == idx)
                self.discard_pile.discard(card)
                agent.hand.remove(card)
                print(f"Agent {self.current_agent} zagral karte {card}")
            CardsEffects(self).realize_effects(card)
        else:
            self.make_action(action, agent)

        done = len(agent.hand) == 0
        reward = 20 if done else 0
        observation = self.get_observation(self.current_agent)
        info = {}

        self.current_agent = (self.current_agent + 1) % NUM_AGENTS

        return observation, reward, done, info

    def get_observation(self, agent_index):
        agent_hand = self.agents[agent_index].hand
        top_card = self.discard_pile.top_card()
        observation = np.zeros(54)

        for card in agent_hand:
            observation[card.index] = 1

        if top_card:
            observation[52] = top_card.index

        if self.requested_value:
            observation[53] = self.requested_value

        return observation

    def valid_action_space(self, agent_index):
        agent_hand = self.agents[agent_index].hand
        top_card = self.discard_pile.top_card()
        valid_actions = []

        for card in agent_hand:
            if top_card.value == 'Q' or card.value == 'Q' or card.value == top_card.value or card.suit == top_card.suit:
                valid_actions.append(card.index)

        value_to_indices = {}
        for card in agent_hand:
            if card.value not in value_to_indices:
                value_to_indices[card.value] = []
            value_to_indices[card.value].append(card.index)

        for indices in value_to_indices.values():
            if len(indices) > 1:
                first_card = next(c for c in agent_hand if c.index == indices[0])
                if top_card.value == 'Q' or first_card.value == 'Q' or first_card.value == top_card.value or first_card.suit == top_card.suit:
                    valid_actions.append(indices)

        if any(card.index in [9, 22, 35, 48] and (card.value == top_card.value or card.suit == top_card.suit) for card in agent_hand):
            for idx in [9, 22, 35, 48]:
                if idx in [card.index for card in agent_hand]:
                    for value_idx in range(48, 54):
                        value_action = (idx, value_idx)
                        valid_actions.append((idx, value_action))

        valid_actions.append("draw")

        return valid_actions

    def make_action(self, action, agent):
        card = next(c for c in agent.hand if c.index == action)
        agent.hand.remove(card)
        self.discard_pile.discard(card)
        CardsEffects(self).realize_effects(card)

    def render(self, mode='human'):
        for agent in self.agents:
            print(f"Agent {agent.agent_id} ma karty: {[str(card) for card in agent.hand]}")
        print(f"Karta na stosie: {self.discard_pile.top_card()}")
        print(f"Rozmiar talii: {len(self.deck.cards)}")


if __name__ == "__main__":
    env = MacauEnv()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agents = [DQNAgent(state_size, action_size) for _ in range(NUM_AGENTS)]
    done = False
    batch_size = 32

    for e in range(NUM_EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        agent_indices = list(range(NUM_AGENTS))

        for time in range(500):
            current_agent_index = env.current_agent
            current_agent = agents[current_agent_index]
            valid_actions = env.valid_action_space(current_agent_index)
            action = current_agent.act(state, valid_actions)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            current_agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(f"Agent {current_agent_index} wygrał w epizodzie {e + 1}/{NUM_EPISODES} po {time + 1} krokach")
                break
            if len(current_agent.memory) > batch_size:
                current_agent.replay(batch_size)

        if (e + 1) % 50 == 0:
            for i, agent in enumerate(agents):
                agent.save(f"macau-dqn-agent-{i}-{e + 1}.h5")

