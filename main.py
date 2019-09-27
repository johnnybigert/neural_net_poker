from pypokerengine.api.game import start_poker, setup_config

from botplayer import BotPlayer
from allinbot import AllInBot
from allinrandomplayer import AllInRandomPlayer
import numpy as np

if __name__ == '__main__':
    stack_log = []
    bot = AllInBot()
    hero = BotPlayer(bot)
    stop = True
    for round in range(1000001):
        opp = AllInRandomPlayer()

        do_print = True if round % 500 == 0 else False
        bot.debug_level = (2 if do_print else 0)
        if do_print:
            print()

        config = setup_config(max_round=1, initial_stack=100, small_blind_amount=5)
        if round % 2 == 0:
            config.register_player(name="hero", algorithm=hero)
        config.register_player(name="opp", algorithm=opp)
        if round % 2 == 1:   # vary the SB
            config.register_player(name="hero", algorithm=hero)
        game_result = start_poker(config, verbose=(2 if do_print else 0))

        stack_log.append([player['stack'] for player in game_result['players'] if player['uuid'] == hero.uuid])

        if round % 10000 == 0:
            print(f'Round {round}, avg. stack: {np.mean(stack_log):3.2}')
            bot.output_predictions()
            stack_log = []
