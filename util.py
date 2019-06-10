
def suit(card):
    c = card[0]
    return {
        "C": 0,
        "D": 1,
        "H": 2,
        "S": 3
    }[c]

def rank(card):
    c = card[1]
    return {
        "A": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
        "7": 7,
        "8": 8,
        "9": 9,
        "T": 10,
        "J": 11,
        "Q": 12,
        "K": 13,
    }[c] - 1


def player_is_all_in(uuid, round_state):
    all_in = [seat for seat in round_state['seats']
                 if seat['uuid'] == uuid and seat['state'] == 'allin']
    return not not all_in


def fold_or_all_in(action, valid_actions):
    if action == 'fold':
        return valid_actions[0]['action'], valid_actions[0]['amount']
    elif action == 'allin':
        return valid_actions[2]['action'], 100
    else:
        raise ValueError(f'action "{action}" is not "fold" or "allin"')
