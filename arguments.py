from dataclasses import dataclass

@dataclass
class Epsilon:
    epsilon_decay_rate = 0.2
    epsilon_start = 1.0
    epsilon_end = 0.02