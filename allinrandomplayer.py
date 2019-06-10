from pypokerengine.players import BasePokerPlayer
import random as rand
import util


class AllInRandomPlayer(BasePokerPlayer):

    def __init__(self):
        pass

    def declare_action(self, valid_actions, hole_card, round_state):
        action = rand.choice(['fold', 'allin'])
        result = util.fold_or_all_in(action, valid_actions)
        return result

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, new_action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass
