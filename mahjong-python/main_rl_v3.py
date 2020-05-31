# -*- coding: utf-8 -*-

import main_rl as base

SessionManagerBase = base.SessionManager


class SessionManager(SessionManagerBase):
    def collect_examples(self, agents, examples):
        examples.extend(e.to_tuple() for agent in agents for es in agent.examples for e in es)

    def create_agents(self):
        (n_agent, p_agent, _) = super().create_agents()[0]

        return (n_agent, p_agent, _), [0, 1, 1, 1]


if __name__ == '__main__':
    base.SessionManager = SessionManager
    base.main()

