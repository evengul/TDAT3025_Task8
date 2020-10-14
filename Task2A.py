# Create a gridworld-environment and solve positions
import pygame
import numpy as np
import random
import sys
from QLearning import QLearning

vec = pygame.math.Vector2
GRID_SIZE = 10
BOX_LENGTH = 500
PLAYER_LENGTH = 40
GRID_WIDTH = 5
MOVE_LENGTH = PLAYER_LENGTH + 2 * GRID_WIDTH  # PLAYER_LENGTH / 2 + 2 * GRID_WIDTH + PLAYER_LENGTH / 2
ZERO_POS = GRID_WIDTH + PLAYER_LENGTH / 2


class Arrow(pygame.sprite.Sprite):
    def __init__(self, x=0, y=0):
        super().__init__()
        self.x = x
        self.y = y
        self.color = (255, 0, 0)


class Target(pygame.sprite.Sprite):
    def __init__(self, x=3, y=4):
        super().__init__()
        self.x = x
        self.y = y
        self.color = (255, 255, 255)


class Player(pygame.sprite.Sprite):
    def __init__(self, x=0, y=0):
        super().__init__()
        self.x = x
        self.y = y
        self.color = (128, 255, 40)

    def move(self, direction):
        if direction == 0 and self.y - 1 > -1:
            self.y -= 1
        elif direction == 1 and self.x - 1 > -1:
            self.x -= 1
        elif direction == 2 and self.y + 1 < GRID_WIDTH:
            self.y += 1
        elif direction == 3 and self.x + 1 < GRID_WIDTH:
            self.x += 1


class GridWorldActionSpace:
    def __init__(self):
        self.n = 4
        self.actions = [0, 1, 2, 3]

    def sample(self):
        return random.choice(self.actions)


class GridWorldObservationSpace:
    def __init__(self):
        self.high = [GRID_SIZE - 1, GRID_SIZE - 1, GRID_SIZE - 1, GRID_SIZE - 1]
        self.low = [0, 0, 0, 0]
        self.n = tuple([10, 10, 10, 10])


class GridWorld:
    def __init__(self):
        self.steps = 0
        self.player = Player()
        self.target = Target()
        self.arrow = Arrow()

        self.action_space = GridWorldActionSpace()
        self.observation_space = GridWorldObservationSpace()

        self.display_surface = pygame.display.set_mode((BOX_LENGTH, BOX_LENGTH))
        pygame.display.set_caption("GridWorld")

    def step(self, action):
        self.steps += 1
        self.player.move(action)
        reward = self.is_rewarded()
        done = reward == 1
        if self.steps > 50:
            done = True
        return (self.player.x, self.player.y, self.target.x, self.target.y), reward, done, ""

    def is_rewarded(self):
        if self.player.x == self.target.x and self.player.y == self.target.y:
            return 1
        else:
            return 0

    def reset(self):
        self.player = Player()
        self.target = Target()
        self.steps = 0
        return [self.player.x, self.player.y, self.target.x, self.target.y]

    def render_box(self, x, y, color):
        surface = pygame.Surface((PLAYER_LENGTH, PLAYER_LENGTH))
        rect = surface.get_rect(center=(int(ZERO_POS + x * MOVE_LENGTH), int(ZERO_POS + y * MOVE_LENGTH)))
        pygame.draw.rect(self.display_surface, color, rect)

    def render_table(self, Q_table):
        for y in range(len(Q_table)):
            for x in range(len(Q_table[y])):
                if np.argmax(Q_table[y][x][3][9]) > 0:
                    color_value = int(round(np.argmax(Q_table[y][x][3][9]) / 4. * 255))
                    self.render_box(x, y, (color_value, color_value, color_value))
                else:
                    self.render_box(x, y, (0, 0, 255))

    def render_target(self):
        self.render_box(self.target.x, self.target.y, self.target.color)

    def render_player(self):
        self.render_box(self.player.x, self.player.y, self.player.color)

    def render(self, Q_table):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        self.display_surface.fill((0, 0, 0))

        self.render_table(Q_table)
        self.render_player()
        self.render_target()

        pygame.display.update()
        pygame.time.wait(250)


agent = QLearning(GridWorld(), buckets=(10, 10, 10, 10))


def run():
    t = 0
    done = False
    current_state = agent.discretize_state(agent.env.reset())
    while not done:
        t += 1
        action = agent.choose_action(current_state)
        agent.env.render(agent.Q_table)
        obs, reward, done, _ = agent.env.step(action)
        new_state = agent.discretize_state(obs)
        current_state = new_state
    return t


agent.train()
agent.plot_learning()

# Used to find the positions the Q-table is actually using
# for y_player in range(len(agent.Q_table)):
#     for x_player in range(len(agent.Q_table[y_player])):
#         for y_target in range(len(agent.Q_table[y_player][x_player])):
#             for x_target in range(len(agent.Q_table[y_player][x_player][y_target])):
#                 if np.argmax(agent.Q_table[y_player][x_player][y_target][x_target]) > 0:
#                     print("(%i,%i,%i,%i)=>%s" % (y_player, x_player, y_target, x_target,
#                                                  np.argmax(agent.Q_table[y_player][x_player][y_target][x_target])))

print("Successful: " + str(run() < 50))









