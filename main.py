import numpy as np
import pandas as pd
import random
import pygame
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam

# 📌 Önce Pygame başlatılıyor
pygame.init()

# Oyun parametreleri
width, height = 360, 360
fps = 30
white, black, red, blue, green = (255, 255, 255), (0, 0, 0), (255, 0, 0), (0, 0, 255), (0, 255, 0)

# 📌 Pygame ekranını oluştur
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("RL GAME")

class Player(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((20, 20))
        self.image.fill(blue)
        self.rect = self.image.get_rect()
        self.rect.centerx, self.rect.bottom = width / 2, height - 1
        self.speedx, self.radius = 0, 10
        pygame.draw.circle(self.image, red, self.rect.center, self.radius)

    def update(self, action):
        keys = pygame.key.get_pressed()  # 📌 Pygame olaylarını düzgün işle
        self.speedx = -4 if keys[pygame.K_LEFT] or action == 0 else 4 if keys[pygame.K_RIGHT] or action == 1 else 0

        self.rect.x += self.speedx
        self.rect.right, self.rect.left = min(self.rect.right, width), max(self.rect.left, 0)

    def getCoordinates(self):
        return self.rect.x, self.rect.y


class Enemy(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((10, 10))
        self.image.fill(red)
        self.rect = self.image.get_rect()
        self.rect.x, self.rect.y = random.randrange(0, width - self.rect.width), random.randrange(2, 6)
        self.radius, self.speedx, self.speedy = 5, 0, 3
        pygame.draw.circle(self.image, white, self.rect.center, self.radius)

    def update(self):
        self.rect.y += self.speedy
        if self.rect.top > height + 10:
            self.rect.x, self.rect.y = random.randrange(0, width - self.rect.width), random.randrange(2, 6)

    def getCoordinates(self):
        return self.rect.x, self.rect.y


class DQLAgent:
    def __init__(self):
        self.state_size, self.action_size = 4, 3
        self.gamma, self.learning_rate = 0.95, 0.001
        self.epsilon, self.epsilon_decay, self.epsilon_min = 1, 0.995, 0.01
        self.memory = deque(maxlen=1000)
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Input(shape=(4,)))
        model.add(Dense(48, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state = np.array(state).reshape(1, 4)
        return random.randrange(self.action_size) if np.random.rand() <= self.epsilon else np.argmax(
            self.model.predict(state)[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state, next_state = np.array(state).reshape(1, 4), np.array(next_state).reshape(1, 4)
            target = reward if done else reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            train_target = self.model.predict(state)
            train_target[0][action] = target
            self.model.fit(state, train_target, verbose=0)

    def adaptiveEGreedy(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)


class Env:
    def __init__(self):
        self.screen = screen  # 📌 Ekranı doğrudan kullan
        self.all_sprite, self.enemy = pygame.sprite.Group(), pygame.sprite.Group()
        self.player = Player()
        self.all_sprite.add(self.player)
        self.m1, self.m2 = Enemy(), Enemy()
        self.enemy.add(self.m1, self.m2)
        self.all_sprite.add(self.m1, self.m2)
        self.reward, self.done, self.total_reward = 0, False, 0
        self.agent = DQLAgent()
        self.clock = pygame.time.Clock()

    def findDistance(self, a, b):
        return a - b

    def step(self, action):
        state_list = []
        self.player.update(action)
        self.m1.update()
        self.m2.update()

        next_player_state, next_m1_state, next_m2_state = self.player.getCoordinates(), self.m1.getCoordinates(), self.m2.getCoordinates()

        state_list += [
            self.findDistance(next_player_state[0], next_m1_state[0]),
            self.findDistance(next_player_state[1], next_m1_state[1]),
            self.findDistance(next_player_state[0], next_m2_state[0]),
            self.findDistance(next_player_state[1], next_m2_state[1])
        ]
        return state_list

    def initialStates(self):
        return self.step(0)

    def run(self):
        state = self.initialStates()
        running, batch_size = True, 24

        while running:
            self.reward = 2
            self.clock.tick(fps)

            # 📌 Pygame olaylarını yönet
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            action = self.agent.act(state)
            next_state = self.step(action)
            self.total_reward += self.reward
            hits = pygame.sprite.spritecollide(self.player, self.enemy, False, pygame.sprite.collide_circle)

            if hits:
                self.reward, self.total_reward = -150, self.total_reward - 150
                self.done, running = True, False
                print("Total Reward:", self.total_reward)

            self.agent.remember(state, action, self.reward, next_state, self.done)
            state = next_state
            self.agent.replay(batch_size)
            self.agent.adaptiveEGreedy()

            self.screen.fill(green)
            self.all_sprite.draw(self.screen)
            pygame.display.flip()

        pygame.quit()


if __name__ == "__main__":
    env = Env()
    t = 0
    while True:
        t += 1
        print("Episode:", t)
        env.run()
