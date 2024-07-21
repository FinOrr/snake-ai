import torch
import random
import numpy as np
from collections import deque
from game import BLOCK_SIZE, SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001              # Learning rate

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0    # Controls randomness
        self.gamma = 0.9    # Discount rate (~0.8 -> 0.9)
        self.memory = deque(maxlen=MAX_MEMORY) # If we exceed the memory,  popleft()
        self.model = Linear_QNet(11, 512, 3) # Hidden layer size can be changed
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma) 

    def get_state(self, game):
        head = game.snake[0]                            # Get snake head location
        point_l = Point(head.x - BLOCK_SIZE, head.y)    # Get the location of the blocks
        point_r = Point(head.x + BLOCK_SIZE, head.y)    # that are surrounding the snake
        point_u = Point(head.x, head.y - BLOCK_SIZE)    # head, U/R/D/L
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT        # Boolean result to indicated where
        dir_r = game.direction == Direction.RIGHT       # the snake is pointing
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight ahead
            (dir_l and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger to the right of snake's direction of travel
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)) or
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)),

            # Danger to the left of the snake's direction of travel
            (dir_l and game.is_collision(point_d)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_d and game.is_collision(point_r)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food Location
            game.food.x < game.head.x,  # Food is further left than the snake on the screen
            game.food.x > game.head.x,  # Food is further right than the snake
            game.food.y < game.head.y,  # Food is further up than the snake
            game.food.y > game.head.y,  # Food is further down than the snake
        ]

        return np.array(state, dtype=int)   # Convert our state list into an array of ints

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # Popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # Returns a list of tuples
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # In the beginning do some random moves: tradeoff between exploration / exploitation
        # We want to make some random moves to explore the environment
        # The better the agent gets, less randomness and more model exploitation is prefereable
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # Get the old state
        state_old = agent.get_state(game)
        # Get the move
        final_move = agent.get_action(state_old)
        # perform the move and get the new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        # Train the short memory for one step
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        # Remember 
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # Train the long memory ("experience replay") using all the previous games
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:      # If we get a new high-score
                record = score
                agent.model.save()
            
            print("Game", agent.n_games, "Score", score, "Record:", record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()