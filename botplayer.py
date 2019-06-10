from pypokerengine.engine.hand_evaluator import HandEvaluator
from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import _pick_unused_card, _fill_community_card, gen_cards
import util
from keras.models import Sequential
from keras.layers import Dense
import numpy
import random as rand



class BotPlayer(BasePokerPlayer):
    def __init__(self, bot):
        super().__init__()

        self.bot = bot
        self.hole_cards = []

        self.seed = 7
        numpy.random.seed(self.seed)

    def declare_action(self, valid_actions, hole_cards, round_state):
        if self.new_game_started:
            self.new_game(hole_cards, round_state)
        hero_is_all_in = self.player_is_all_in(self.uuid, round_state)
        opponent_is_all_in = self.player_is_all_in(self.opp_uuid, round_state)
        action_number = len(round_state['action_histories']['preflop']) - 2
        action = self.bot.heros_turn(action_number, hero_is_all_in, opponent_is_all_in)
        result = util.fold_or_all_in(action, valid_actions)
        return result

    def hero_is_big_blind(self, round_state):
        big_blind = [action for action in round_state['action_histories']['preflop']\
                     if action['action'] == 'BIGBLIND' and action['uuid'] == self.uuid]
        return not not big_blind

    def player_is_all_in(self, uuid, round_state):
        all_in = [seat for seat in round_state['seats']
                     if seat['uuid'] == uuid and seat['state'] == 'allin']
        return not not all_in

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_cards, seats):
        self.hole_cards = hole_cards
        self.opp_uuid = [seat['uuid'] for seat in seats if seat['name'] == 'opp'][0]
        self.new_game_started = True
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        if self.new_game_started:
            self.new_game(self.hole_cards, round_state)
        if self.is_opponents_turn(round_state):
            hero_is_all_in = self.player_is_all_in(self.uuid, round_state)
            opponent_is_all_in = self.player_is_all_in(self.opp_uuid, round_state)
            action_number = len(round_state['action_histories']['preflop']) - 3   # action after SB and BB should have index 0
            self.bot.opponents_turn(action_number, hero_is_all_in, opponent_is_all_in)
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        is_winner = self.uuid in [item['uuid'] for item in winners]
        main_pot_size = round_state['pot']['main']['amount']
        hero_can_win_side_pots = \
            sum(side_pot['amount'] for side_pot in round_state['pot']['side'] if self.uuid in side_pot['eligibles'])
        opp_can_win_side_pots = \
            sum(side_pot['amount'] for side_pot in round_state['pot']['side'] if self.opp_uuid in side_pot['eligibles'])
        hero_paid = self.get_paid_sum(round_state, self.uuid)
        opp_paid = self.get_paid_sum(round_state, self.opp_uuid)
        if is_winner:
            profit = main_pot_size + hero_can_win_side_pots - hero_paid
        else:
            profit = -(main_pot_size + opp_can_win_side_pots - opp_paid)

        self.bot.game_ended(profit)

    def get_paid_sum(self, round_state, uuid):
        return self.get_paid_sum_round(round_state, 'preflop', uuid) + \
               self.get_paid_sum_round(round_state, 'flop', uuid) + \
               self.get_paid_sum_round(round_state, 'turn', uuid) + \
               self.get_paid_sum_round(round_state, 'river', uuid)

    def get_paid_sum_round(self, round_state, round, uuid):
        actions = round_state['action_histories'].get(round, [])
        return sum([action.get('paid', action.get('amount', 0))       # 'amount' is for blinds, 'paid' for all other actions
                    for action in actions
                    if action['uuid'] == uuid])

    def new_game(self, hole_cards, round_state):
        posted_big_blind = self.hero_is_big_blind(round_state)
        self.bot.new_game(hole_cards, posted_big_blind)
        self.new_game_started = False

    def is_opponents_turn(self, round_state):
        result = round_state['action_histories']['preflop'][-1]['uuid'] == self.opp_uuid
        return result

