# AlphaPace
AI to play Mancala - alpha-beta pruning guided search algorithm which evaluates the possibility of a win based on the board states within the branch using a Recurrent Neural Network.

## Student Data
David Penz, 11703497

Giulio Pace, 11835706

## Info on Architecture
The initial idea for the agent was an architectur revolving around optimising the alpha-beta pruning algorithm. For this purpose, we implemented alpha-beta pruning from scratch and tried to enhance the algorithm with the following optimisations:
- iterative deepening search - starting with a shallow depth to speed up the initial alpha-beta pruning, but continuing to deepen the search tree after each iteration until the set time limit is met (idea: explore more in into the future if the time allows)
- selection criteria - in case two or more nodes will result an identical heuristic value, both starting moves will be added to a list and then selected randomly by the agent when deciding where to move
- more elaborate heuristic method: Similarly to stockfish centipawn evaluation, we try to not only evaluate OwnDepot - EnemyDepot, but we implemented a heuristic that keeps into account the position of all stones. 

Unfortunately, this initial idea (see "AlphaPace_old.java" as reference for the code) was very weak, as even the pre-built alpha-beta pruning agent of the provided engine would beat it easily. Therefore, we decided to use the pre-built alpha-beta pruning agent as skeleton and focus on the following ideas:

### RNN as Heuristic
As core concept of our agent, we we took inspiration from DeepMind's AlphaGo / AlphaZero. Obviously our agent is way less complex and uses alpha-beta pruning instead of Monte Carlo Tree Search. 
The evaluation is performed by a simple RNN: we feed a board state to the network, that will return its prediction on the probability of victory in the given position.

### SelfPlay for training
We created a method "selfPlay" within the agents class in order to simulate games without starting the GUI and to store the game states. The method creates a copy of the current game and runs multiple simulations where the agents plays against itself. After a batch of games, the neural network is trained on the simulated games and evaluated. This process has been repeated six times with batch of games that vary in size from 300 to 1000.

### Selection Criteria
To include some randomness in our algorithm we introduced the parameter delta. When we evaluate a position, we add it in the pool of "best positions" if it is in the range of (best - delta) - (best + delta). If it is better than best + delta it becomes the new best.


## Info on Setup
The agent itself can be found as "AlphaPace.java" or "AlphaPace.jar" (class name: at.pwd.mancala.AlphaPace).

*In case the .jar file is missing the required model file or dependencies:*
The file "AlphaPace.h5" represents the trained neural network. This file has to be copied into src/main/resources/ prior to starting the engine. 

Additionaly, the Java libraries Deeplearning4J and the connected library ND4J need to be added to the dependencies. Below you can find the additional dependencies taken from the build.gradle file:
```
dependencies {
    compile "org.deeplearning4j:deeplearning4j-core:1.0.0-beta2"
    compile "org.deeplearning4j:deeplearning4j-modelimport:1.0.0-beta2"
    compile "org.nd4j:nd4j-native-platform:1.0.0-beta2"
    compile "org.eclipse.jetty:jetty-server:9.4.9.v20180320"
    compile "com.google.cloud.dataflow:google-cloud-dataflow-java-sdk-all:2.2.0"
    compile "org.slf4j:slf4j-api:1.7.5"
    compile "org.slf4j:slf4j-log4j12:1.7.5"
}
```

## Hardware Specs
### Computer 1
Ubuntu 18.04.3 LTS 64bit
RAM 13,6 GiB
CPU AMD® Ryzen 5 pro 3500u w/ radeon vega mobile gfx × 8 

### Computer 2
Windows 10 Pro
RAM 8 GB
CPU Intel Core i7-6600 CPU 2.6 GHz

### Computer 3
Windows 10 Pro
RAM 16 GB
CPU Intel Core i7 3.2 GHz
GPU Nvidia GTX 1070
