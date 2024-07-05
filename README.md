# Average-reward reinforcement learning in two-player two-action games

This repository contains the Mathematica code used for my Bachelor Assignment. It is largely based on the code used in [[1]](#1).

The files used for the results are seperated per game. They are found in PrisonersDilemma.nb, StagHunt.nb and Snowdrift.nb.

ExactAverageReward.nb contains the approach of computing the average reward exactly instead of estimating it. It is applied in the prisoner's dilemma, where the resulting transition conditions where found to be the same as in PrisonersDilemma.nb.

simulation.py is the failed attempt to simulate the environment with a finite batch size, as mentioned in the conclusion of the article. I invite you to try to fix it. 


## References
<a id="1">[1]</a> 
Janusz M. Meylahn and Lars Janssen.
Limiting Dynamics for Q-Learning with Memory One in Symmetric Two-Player, Two-Action Games.
Complexity, 2022:e4830491, November 2022. Publisher: Hindawi
