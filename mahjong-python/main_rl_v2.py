# -*- coding: utf-8 -*-

import main_rl as base

SessionManagerBase = base.SessionManager


class SessionManager(SessionManagerBase):
    def collect_examples(self, agents, examples):
        examples.extend(e.to_tuple() for agent in agents for es in agent.examples for e in es)

    def create_agents(self):
        (n_agent, p_agent, fixed_agent), _ = super().create_agents()

        return (n_agent, p_agent, fixed_agent), [0, 1, 2, 2]


if __name__ == '__main__':
    base.SessionManager = SessionManager
    base.main()
