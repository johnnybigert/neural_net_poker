import operator

from util import *
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from keras.layers import InputLayer



class AllInBot:
    def __init__(self):
        super().__init__()

        self.gamma = 0.95
        self.start_eps = 0.5
        self.eps_decay = 0.99999
        self.current_state = None
        self.hole_cards = []
        self.hero_posted_big_blind = False
        self.eps = 0
        self.batch_size = 3000

        self.new_game(['S5', 'DQ'], True)  # random cards to get input size

        # examples to determine input/output size
        self.input_size = self._encode_input(self.hole_cards, self.current_state).shape[1]
        self.output_size = self._encode_output(100).shape[1]
        self.batch_input = np.zeros(shape=(self.batch_size, self.input_size))
        self.batch_output = np.zeros(shape=(self.batch_size, self.output_size))
        self.batch_counter = 0

        self.model = Sequential()
        self.model.add(Dense(10, input_dim=self.input_size, activation='relu'))
        self.model.add(Dense(10, activation='relu'))
        self.model.add(Dense(self.output_size, activation='sigmoid'))
        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])

    def new_game(self, hole_cards, hero_posted_big_blind):
        self.current_state = ([0, 0, 0], [0, 0, 0])
        self.hole_cards = hole_cards
        self.hero_posted_big_blind = hero_posted_big_blind
        self.eps = self.start_eps

    def heros_turn(self, action_number, hero_is_all_in, opponent_is_all_in):
        if self._is_randomized_action():
            a = np.random.randint(0, 2)
        else:
            a = self._predict_action(self.hole_cards, self.current_state)
        self._update_current_state(action_number, a)
        #print(f'update hero {action_number} {a}')
        return ('fold', 'allin')[a]

    def opponents_turn(self, action_number, hero_is_all_in, opponent_is_all_in):
        self._update_current_state(action_number, 1 if opponent_is_all_in else 0)
        #print(f'update opp {action_number} {1 if opponent_is_all_in else 0}')
        pass

    def game_ended(self, profit):
        self._train_model(self.hole_cards, self.current_state, profit)
        if self._heros_small_blind_move_was_all_in(self.current_state):
            small_blind_state = ([1, 0, 0], [1, 0, 0])
            self._train_model(self.hole_cards, small_blind_state, profit)  # to be able to predict SB action
        pass



    def _store_game_info(self, hole_cards, hero_is_big_blind):
        self.hole_cards = hole_cards
        self.hero_is_big_blind = hero_is_big_blind

    def _is_randomized_action(self):
        self.eps *= self.eps_decay
        return np.random.random() < self.eps

    def _predict_action(self, hole_cards, current_state):
        fold_state, all_in_state = self._fold_and_all_in_states(current_state)

        fold_input = self._encode_input(hole_cards, fold_state)
        fold_prediction = self.model.predict(fold_input)[0][0]
        fold_profit = self._decode_output(fold_prediction)
        if self.batch_counter % 100 == 0:
            print(f'predict - hole cards: {hole_cards[0]} {hole_cards[1]}, state: {fold_state} output: {fold_prediction:.3} (${fold_profit:.3})')

        all_in_input = self._encode_input(hole_cards, all_in_state)
        all_in_prediction = self.model.predict(all_in_input)[0][0]
        all_in_profit = self._decode_output(all_in_prediction)
        if self.batch_counter % 100 == 0:
            print(f'predict - hole cards: {hole_cards[0]} {hole_cards[1]}, state: {all_in_state} output: {all_in_prediction:.3} (${all_in_profit:.3})')

        if all_in_profit > fold_profit:
            return 1   # all-in
        else:
            return 0   # fold

    def _train_model(self, hole_cards, state, profit):
        input = self._encode_input(hole_cards, state)
        output = self._encode_output(profit)
        self.batch_input[self.batch_counter] = input
        self.batch_output[self.batch_counter] = output
        self.batch_counter += 1
        if self.batch_counter % 100 == 0:
            print(f'train {self.batch_counter} - hole cards: {hole_cards[0]} {hole_cards[1]}, state: {state} '
                  f'output: {output[0][0]:.3} (${profit})')
        if self.batch_counter == self.batch_size:
            self.model.fit(self.batch_input, self.batch_output, epochs=100, verbose=2)
            self.batch_counter = 0

    def _update_current_state(self, action_number, a):
        used, actions = self.current_state
        i = action_number + (1 if self.hero_posted_big_blind else 0)
        used[i] = 1
        actions[i] = a
        #print(self.current_state)

    def _fold_and_all_in_states(self, state):
        used_list = enumerate(state[0])
        indexes = [index for (index, value) in used_list if value == 1]
        if indexes:
            i = max(indexes) + 1
        else:
            i = 0
        used = [] + state[0]
        fold_actions = [] + state[1]
        all_in_actions = [] + state[1]
        used[i] = 1
        fold_actions[i] = 0
        all_in_actions[i] = 1
        return (used, fold_actions), (used, all_in_actions)

    def _heros_small_blind_move_was_all_in(self, state):
        if self.hero_posted_big_blind:
            return False
        actions = state[1]
        hero_sb_action = actions[0]
        return hero_sb_action == 1   # all-in

    def _encode_input(self, hole_cards, state):
        result = self._encode_hole_cards(hole_cards)
        result += state[0]  # used
        result += state[1]  # actions
        return np.array([result])

    def _encode_output(self, profit):
        result = (profit + 100) / 200
        return np.array([[result]])

    def _decode_output(self, prediction):
        result = prediction * 200 - 100
        return result

    def _encode_hole_cards(self, hole_cards):
        # For example, S5 DQ
        #         A  2  3  4  5  6  7  8  9  T  J  Q  K     C  S  D  H
        # return [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0] + [0, 1, 0, 0] + \
        #        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0] + [0, 0, 1, 0]

        ranks0 = [0] * 13
        suits0 = [0] * 4
        ranks1 = [0] * 13
        suits1 = [0] * 4
        r0 = rank(hole_cards[0])
        s0 = suit(hole_cards[0])
        r1 = rank(hole_cards[1])
        s1 = suit(hole_cards[1])
        ranks0[r0] = 1
        suits0[s0] = 1
        ranks1[r1] = 1
        suits1[s1] = 1

        result = ranks0 + suits0 + ranks1 + suits1
        return result

    def output_predictions(self):
        self.output_bigblind_prediction(True)
        self.output_bigblind_prediction(False)

    def output_bigblind_prediction(self, hero_is_big_blind):
        self.output_prediction(hero_is_big_blind, ['SA', 'DA'])
        self.output_prediction(hero_is_big_blind, ['CA', 'HA'])
        self.output_prediction(hero_is_big_blind, ['SA', 'SK'])
        self.output_prediction(hero_is_big_blind, ['SK', 'DK'])
        self.output_prediction(hero_is_big_blind, ['SQ', 'DQ'])
        self.output_prediction(hero_is_big_blind, ['SA', 'ST'])
        self.output_prediction(hero_is_big_blind, ['SK', 'DT'])
        self.output_prediction(hero_is_big_blind, ['ST', 'DT'])
        self.output_prediction(hero_is_big_blind, ['ST', 'D9'])
        self.output_prediction(hero_is_big_blind, ['S9', 'D7'])
        self.output_prediction(hero_is_big_blind, ['S7', 'D7'])
        self.output_prediction(hero_is_big_blind, ['S7', 'D2'])
        self.output_prediction(hero_is_big_blind, ['S4', 'D4'])
        self.output_prediction(hero_is_big_blind, ['S4', 'D3'])
        self.output_prediction(hero_is_big_blind, ['S4', 'D2'])
        self.output_prediction(hero_is_big_blind, ['S2', 'D2'])

    def output_prediction(self, hero_is_big_blind, hole_cards):
        if hero_is_big_blind:
            state = [[0, 1, 0], [0, 1, 0]]
        else:
            state = [[0, 0, 0], [0, 0, 0]]

        fold_state, all_in_state = self._fold_and_all_in_states(state)

        fold_input = self._encode_input(hole_cards, fold_state)
        fold_prediction = self.model.predict(fold_input)[0][0]
        fold_profit = self._decode_output(fold_prediction)
        print(f'big blind: {hero_is_big_blind}, hole cards: {hole_cards[0]} {hole_cards[1]}, state: {fold_state} fold   output: {fold_prediction:.3} (${fold_profit:.3})')

        all_in_input = self._encode_input(hole_cards, all_in_state)
        all_in_prediction = self.model.predict(all_in_input)[0][0]
        all_in_profit = self._decode_output(all_in_prediction)
        print(f'big blind: {hero_is_big_blind}, hole cards: {hole_cards[0]} {hole_cards[1]}, state: {all_in_state} all-in output: {all_in_prediction:.3} (${all_in_profit:.3})')

