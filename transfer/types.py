import typing

from dataclasses import dataclass


class AgentMetadata:
    id: int
    category: str
    outputSize: int
    alive: bool
    netWorth: float


class Reward:

    # The one sending this reward
    senderId: int

    # The intended recipient of this reward
    receiverId: int

    # The timestep that this reward is associated with
    timestep: int

    # The net amount that this reward consists of
    amount: float

    # Any gradients associated w/ this reward
    gradients: any


class Output:
    output: any
    rewards: typing.List[Reward]


@dataclass
class PresentContext:
    # All of the outputs for all of the agents from the last timestep
    agentOutput: typing.Dict[int, Output]

    # Rewards, keyed by the receiver
    rewards: typing.Dict[int, typing.List[Reward]]

    # Metadata for all agents
    agentMeta: typing.Dict[int, AgentMetadata]

    # Timestamp that just increments every iteration
    currentTime: int


class Agent:
    id: int

    def run(self, ctx: PresentContext) -> Output:
        pass


class World:
    ctx: PresentContext
    agents: typing.Dict[int, Agent]

    def runStep(self, ctx: PresentContext, agents: typing.Dict[int, Agent]):

        # Compute outputs
        outputs = {x: agents[x].run(ctx) for x in agents}

        # Invert Rewards
        rewards: typing.Dict[int, typing.List[Reward]] = {}
        for x in outputs:
            for reward in outputs[x].rewards:
                receiverId = reward.receiverId
                l = rewards.get(receiverId, [])
                l.append(reward)
                rewards[receiverId] = l

        # Scale agents
        agentMeta = self.scaleAgents(ctx, rewards)
        newCtx = PresentContext(
            outputs, rewards, agentMeta, ctx.currentTime + 1)

        return newCtx

    def scaleAgents(self, ctx: PresentContext, rewards: typing.Dict[int, typing.List[Reward]]) -> AgentMetadata:
        return ctx.agentMeta
