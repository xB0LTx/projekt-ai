import random
import gym
from gym import spaces

CARDS_PER_AGENT = 6
NUM_AGENTS = 4
NUM_EPISODES = 1000


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

        current_agent_id = self.macau_env.current_agent
        agent_hand = self.macau_env.agents[current_agent_id].hand

        value_to_indices = {}
        for card in agent_hand:
            if card.value not in value_to_indices:
                value_to_indices[card.value] = []
            value_to_indices[card.value].append(card.index)

        request_value = max(value_to_indices, key=lambda k: len(value_to_indices[k]))

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

        self.action_space = spaces.Discrete(53 + 6)  # 52 karty + 1 dobranie karty + 6 wartosci (5-10)
        self.observation_space = spaces.Dict({
            'agent_hand': spaces.MultiDiscrete([52] * 52),
            'top_card': spaces.MultiDiscrete([52]),
            'opponent_hand_size': spaces.Discrete(52),
            'deck_size': spaces.Discrete(52)
        })

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
            #CardsEffects(self).value_request()
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

        return {
            'agent_hand': [(card.index, card.value, card.suit) for card in agent_hand],
            'top_card': (top_card.index, top_card.value, top_card.suit) if top_card else None,
            'opponent_hand_size': len(self.agents[(agent_index + 1) % NUM_AGENTS].hand),
            'deck_size': len(self.deck.cards)
        }

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
                    for value_action in range(53, 59):
                        valid_actions.append((idx, value_action))

        valid_actions.append("draw")

        return valid_actions

    def make_action(self, action, agent):
        if action == "draw":
            drawn_card = self.deck.draw_card(self.discard_pile)
            agent.hand.append(drawn_card)
            print(f"Agent {self.current_agent} dobral karte {drawn_card}")
        else:
            if isinstance(action, list):
                for idx in action:
                    card = next(c for c in agent.hand if c.index == idx)
                    self.discard_pile.discard(card)
                    agent.hand.remove(card)
                    print(f"Agent {self.current_agent} zagral karte {card}")
            else:
                card = next(c for c in agent.hand if c.index == action)
                self.discard_pile.discard(card)
                agent.hand.remove(card)
                print(f"Agent {self.current_agent} zagral karte {card}")

            CardsEffects(self).realize_effects(card)


if __name__ == "__main__":
    env = MacauEnv()
    results = env.reset()
    done = False

    for i in range(1):
        print(f"Episode {i}\n")
        while not done:
            current_agent = env.current_agent
            valid_action = env.valid_action_space(current_agent)

            action = random.choice(valid_action)

            results = env.step(action)

            observation = results[0]
            rewards = results[1]
            done = results[2]
            info = results[3]

            print(f"\nReka agenta {current_agent}: {observation['agent_hand']} ")
            print(f"Karta na srodku: {observation['top_card']}\n___________________________________________\n\n")

            if done:
                obs = env.reset()



