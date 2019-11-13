import h5py

from dlgo import rl
from dlgo import agent
from dlgo.gotypes import Player
from dlgo.rl.simulate import simulate_game

agent1 = agent.load_policy_agent(h5py.File('../agents/agent1.h5'))
agent2 = agent.load_policy_agent(h5py.File('../agents/agent2.h5'))
collector1 = rl.ExperienceCollector()
collector2 = rl.ExperienceCollector()
agent1.set_collector(collector1)
agent2.set_collector(collector2)

for i in range(1):
    collector1.begin_episode()
    collector2.begin_episode()

    game_record = simulate_game(agent1, agent2)
    if game_record.winner == Player.black:
        collector1.complete_episode(reward=1)
        collector2.complete_episode(reward=-1)
    else:
        collector2.complete_episode(reward=1)
        collector1.complete_episode(reward=-1)

experience = rl.combine_experience([collector1, collector2])
with h5py.File('../experience.h5', 'w') as experience_outf:
    experience.serialize(experience_outf)
