import sys
import os
import random  # You were missing this import

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_agent import BaseAgent

class simpleAgent(BaseAgent):
    def __init__(self, server_url="http://localhost:5000", **kwargs):
        super().__init__(server_url, "simpleAgent", **kwargs)
        
        # 💡 Define the action map HERE, inside __init__
        self.action_map = {
            0: self.up,
            1: self.down,
            2: self.left,
            3: self.right,
        }

    def get_strategy_description(self):
        return "If the current spot is dirty, clean it. Otherwise, move in a random direction."
    
    def think(self):
        if not self.is_connected():
            return False
        
        perception = self.get_perception()
        if not perception or perception.get('is_finished', True):
            return False
        
        # Your logic here
        if perception.get('is_dirty', False):
            return self.suck()
        else:
            random_key = random.randint(0, 3)
            action_to_perform = self.action_map[random_key]
            return action_to_perform()