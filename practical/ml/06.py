import numpy as np
import random

class TicTacToe:
    def __init__(self):
        self.board = [' '] * 9 
        self.current_player = 'X'
    def print_board(self):
        for i in range(0, 9, 3):
            print(" | ".join(self.board[i:i+3]))
            if i < 6:
                print("---------")
    def make_move(self, move):
        if self.board[move] == ' ':
            self.board[move] = self.current_player
            self.current_player = 'O' if self.current_player == 'X' else 'X'
            return True
        return False
    def is_winner(self, player):
        winning_combinations = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
                                (0, 3, 6), (1, 4, 7), (2, 5, 8),
                                (0, 4, 8), (2, 4, 6)]

        for combo in winning_combinations:
            if all(self.board[i] == player for i in combo):
                return True
        return False
    def is_draw(self):
        return ' ' not in self.board
# Q-learning agent
class QLearningAgent:
    def __init__(self, epsilon=0.1, alpha=0.3, gamma=0.9):
        self.q_values = {}
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
    def get_q_value(self, state, action):
        return self.q_values.get((state, action), 0.0)
    def update(self, state, action, reward, next_state):
        max_q_next = max(self.get_q_value(next_state, a) for a in range(9))
        self.q_values[(state, action)] = (1 - self.alpha) * self.get_q_value(state, action) + self.alpha * (reward + self.gamma * max_q_next)
    def choose_action(self, state, game):
        if random.random() < self.epsilon:
            return random.choice([i for i in range(9) if game.board[i] == ' '])
        else:
            best_actions = []
            best_q_value = -float("inf")
            for action in range(9):
                if game.board[action] == ' ':
                    q_value = self.get_q_value(state, action)
                    if q_value > best_q_value:
                        best_actions = [action]
                        best_q_value = q_value
                    elif q_value == best_q_value:
                        best_actions.append(action)
            return random.choice(best_actions)
# Training the agent (you can adjust the number of episodes)
def train(agent, episodes=10000):
    for _ in range(episodes):
        game = TicTacToe()
        state = tuple(game.board)
        while not game.is_draw() and not game.is_winner('X') and not game.is_winner('O'):
            action = agent.choose_action(state, game)
            game.make_move(action)
            next_state = tuple(game.board)
            if game.is_winner('X'):
                agent.update(state, action, 1, next_state)
            elif game.is_winner('O'):
                agent.update(state, action, -1, next_state)
            else:
                agent.update(state, action, 0, next_state)
            state = next_state
# Test the agent
def test(agent):
    game = TicTacToe()
    state = tuple(game.board)
    while not game.is_draw() and not game.is_winner('X') and not game.is_winner('O'):
        game.print_board()
        action = agent.choose_action(state, game)
        print(f"Player {game.current_player} chooses position {action}")
        game.make_move(action)
        state = tuple(game.board)
    game.print_board()
    if game.is_winner('X'):
        print("X wins!")
    elif game.is_winner('O'):
        print("O wins!")
    else:
        print("It's a draw!")
if __name__ == "__main__":
    agent = QLearningAgent()
    train(agent)
    test(agent)
