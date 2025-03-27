import random
import re
from collections import defaultdict, Counter

class MarkovChatbot:
    def __init__(self, order=2):
        self.order = order
        self.model = defaultdict(Counter)

    def tokenize(self, text):
        return re.findall(r"\w+|[^\w\s]", text.lower())

    def train(self, filename):
        with open(filename, "r", encoding="utf-8") as file:
            lines = file.readlines()

        for line in lines:
            words = self.tokenize(line)
            for i in range(len(words) - self.order):
                key = tuple(words[i:i+self.order])
                next_word = words[i+self.order]
                self.model[key][next_word] += 1

    def generate_response(self, seed_text, max_words=20):
        seed_tokens = self.tokenize(seed_text)
        key = tuple(seed_tokens[-self.order:]) if len(seed_tokens) >= self.order else random.choice(list(self.model.keys()))

        response = list(key)
        for _ in range(max_words):
            if key in self.model:
                next_word = random.choices(
                    list(self.model[key].keys()), 
                    weights=list(self.model[key].values())
                )[0]
                response.append(next_word)
                key = tuple(response[-self.order:])
            else:
                break
        return " ".join(response)

class ContextualMarkovChatbot(MarkovChatbot):
    def __init__(self, order=2, context_size=3):
        super().__init__(order)
        self.context = [] 

    def generate_response(self, seed_text, max_words=20):
        self.context.append(seed_text)
        if len(self.context) > 3:  
            self.context.pop(0)
        
        full_seed = " ".join(self.context[-self.order:])  
        return super().generate_response(full_seed, max_words)
