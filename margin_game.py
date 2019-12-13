import pandas as pd

print('Добро пожаловать в игру Margin Game!')

print('Введите число игроков:')
n_players = int(input())

# print("Выберите режим игры: закрытый/открытый/стандартный")
# game_type = input()

print('Введите число раундов для игры:')
n_rounds = int(input())

print('Введите начальный капитал:')
init_money = input()
init_money = 10.0 if init_money.strip() == '' else float(init_money)


players_names_dict = {}
for i in range(1, n_players+1):
    print(f'Введите имя {i}ого игрока (название {i}ой команды):')
    players_names_dict[i] = input()

players_stats = {f"Раунд {i}": {name: {'Денег после прошлого раунда': None, 'Текущее решение': None} 
                 for _, name in players_names_dict.items()} for i in range(1, n_rounds+2)}
players_stats['Раунд 1'] = {name: {'Денег после прошлого раунда': init_money, 'Текущее решение': None} 
                 for _, name in players_names_dict.items()}

def sector_A(n_players: int, money: float) -> float:
    return 6/(n_players)*money

def sector_B(money: float, interest_rate: int=0.1) -> float:
    """
    interest_rate: ставка процента
    """
    return money*(1 + interest_rate)

sectors_dict = {
    'sector_A': sector_A,
    'sector_B': sector_B
}

opportunities_choices = [key for key in sectors_dict]

for round_number in range(1, n_rounds+1):
    print(f'Раунд {round_number}')
    n_current_sector_A_players = 0
    for player in players_stats[f'Раунд {round_number}']:
        print(f'Введите решение для игрока {player}: (Варианты: {opportunities_choices})')
        players_stats[f"Раунд {round_number}"][player]['Текущее решение'] = input()
        if players_stats[f"Раунд {round_number}"][player]['Текущее решение'] == 'sector_A':
            n_current_sector_A_players += 1
    
    for player in players_stats[f'Раунд {round_number}']:
        
        if players_stats[f"Раунд {round_number}"][player]['Текущее решение'] == 'sector_A':
            players_stats[f"Раунд {round_number+1}"][player]['Денег после прошлого раунда'] = sector_A(
                n_players=n_current_sector_A_players,
                money=players_stats[f"Раунд {round_number}"][player]['Денег после прошлого раунда']
            )
        
        elif players_stats[f"Раунд {round_number}"][player]['Текущее решение'] == 'sector_B':
            players_stats[f"Раунд {round_number+1}"][player]['Денег после прошлого раунда'] = sector_B(
                money=players_stats[f"Раунд {round_number}"][player]['Денег после прошлого раунда']
            )
    print('Статистика по данному раунду:')
    print()
    print((pd.DataFrame(players_stats[f'Раунд {round_number+1}']).T))
    print()
