import abc
from habitat.sims.habitat_simulator.actions import HabitatSimActions

# create a base agent class that can be used to create other agents
class Agent(abc.ABC):
    @abc.abstractmethod
    def __init__(self, sim, agent_id=0) -> None:
        self.sim = sim
        self.agent_id = agent_id

    @abc.abstractmethod
    def reset(self) -> None: 
        raise NotImplementedError("Subclass must implement abstract method")

    @abc.abstractmethod
    def act(self, observations) -> HabitatSimActions:
        raise NotImplementedError("Subclass must implement abstract method")
