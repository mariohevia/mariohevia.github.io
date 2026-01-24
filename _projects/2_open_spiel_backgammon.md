---
layout: page
title: "OpenSpiel"
description: Extended the Backgammon game’s observation tensor to include dice values,
img: assets/img/project_images/open_spiel_backgammon.png
importance: 1
category: Open Source (Contributions)
---

Added the missing dice values to the Backgammon observation tensor in OpenSpiel. Before this contribution, learning algorithms did not have access to the full game state, meaning they could not actually learn how to play Backgammon or model its rules, only choose among valid moves. This fix enabled genuine learning of the game and was later used to support coevolutionary experiments with PushGP during my work as a Research Fellow at the University of Birmingham.

<div class="row">
    <div style="width: 40%; margin: 0 auto;">
    {% include figure.html
       path="assets/img/project_images/open_spiel_backgammon.png"
       title="FTIR QA App"
       class="img-fluid rounded z-depth-1"
    %}
</div>
</div>

### **Contribution details**

The [OpenSpiel repository](https://github.com/google-deepmind/open_spiel) (DeepMind’s reinforcement-learning games framework) is a research-oriented framework providing environments and algorithms for reinforcement learning in games. It supports many games, including Backgammon, where agents interact via encoded state vectors used for training and evaluation.

This contribution extends the Backgammon environment in OpenSpiel by incorporating the dice values into the game’s observation tensor. In the previous version of the environment, the observation vector omitted the dice, so learning algorithms received an incomplete game state. As a result, reinforcement-learning agents could not learn the actual rules or dynamics of Backgammon: they could only pick from the set of valid moves at each moment, without any ability to infer why those moves were valid or how dice outcomes shaped the game. There was no practical way to train an agent to play Backgammon or to understand the game mechanics.

The contribution modifies the C++ implementation of Backgammon by appending the two dice values to the observation tensor and updating the associated constants and documentation to reflect the expanded encoding. This ensures that algorithms interacting with the environment receive a complete representation of the state and can learn policies that depend on dice outcomes.

The commit affects four files with +63 insertions and −57 deletions, specifically::
- Backgammon state encoding (`open_spiel/games/backgammon.cc`): two additional entries are appended to the observation vector representing the current dice values.
- Header updates (`open_spiel/games/backgammon.h`): the constant representing the size of the state encoding is increased to account for the two new dice entries, and corresponding documentation comments are adjusted to reflect the new encoding structure.
- Updates to the associated unit tests to reflect the new observation specification.

This fix made it possible to use OpenSpiel’s Backgammon environment for genuine learning experiments. In my Research Fellow role at the University of Birmingham, this extended version was used to test coevolutionary algorithms, including PushGP, enabling the study of agents that learn to play Backgammon directly from the now-complete game state.
