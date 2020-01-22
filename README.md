# AlphaPace
AI to play Mancala - alpha-beta pruning guided search algorithm which evaluates the possibility of a win based on the board states within the branch using a simple RNN.

## Student Data
David Penz, 11703497

Giulio Pace, 11835706

## Info on Architecture
The initial idea for the agent was an architectur revolving around optimising the alpha-beta pruning algorithm. For this purpose, we implemented alpha-beta pruning from scratch and tried to enhance the algorithm with the following optimisations:
- iterative deepening search
starting with a shallow depth to speed up the initial alpha-beta pruning, but continuing to deepen the search tree after each iteration until the set time limit is met (idea: explore more in into the future if the time allows)
- selection criteria
in case two or more nodes will result an identical heuristic value, both starting moves will be added to a list and then selected randomly by the agent when deciding where to move
- heuristic optimisation

Unfortunately, this initial idea (see "AlphaPace_old.java" as reference for the code) turned out to not work properly when chosing the same heuristic (own depot - enemy depot) as in the pre-built alpha-beta pruning agent of the provided engine. Therefore, we decided to use the pre-built alpha-beta pruning agent as skeleton and focus on the following two criteria:

### Heuristic Optimisation
As core concept of our agent, we decided to go for a (kinda) similar idea as DeepMind's AlphaGo / AlphaZero application. Obviously, our agent is way less complex as it uses alpha-beta pruning instead of Monte Carlo Tree Search and also just one very simple RNN for evaluating the board states within the search. We created a method "selfPlay" within the agents class (as we could not find another way to simulate games without starting the GUI) which creates a copy of the current game and runs multiple simulations of agent vs agent (thus, the name selfPlay). Afterwards, the neural network is being trained on all simulated games and evaluated. Then the whole process starts over again and again and again ...

### Selection Criteria
As we now use a neural network to predict the outcome of the game based on the board states (0 for loss - 1 for win), we had to come up with another selection criteria as the heuristic values are very unlikely to be the same. Therefore, we introduced the parameter DELTA, which acts as an allowed deviation from the alpha value where a move is considered to be of "equal value".

## Info on Setup
The agent itself can be found as "AlphaPace.java" or "AlphaPace.jar" (class name: at.pwd.mancala.AlphaPace).

*In case the .jar file is missing the required model file or dependencies:*
The file "AlphaPace.h5" represents the trained neural network. This file has to be copied into src/main/resources/ prior to starting the engine. 

Additionaly, the Java libraries Deeplearning4J and the connected library ND4J need to be added to the dependencies. Below you can find the additional dependencies taken from the build.gradle file:
'''
dependencies {
    compile "org.deeplearning4j:deeplearning4j-core:1.0.0-beta2"
    compile "org.deeplearning4j:deeplearning4j-modelimport:1.0.0-beta2"
    compile "org.nd4j:nd4j-native-platform:1.0.0-beta2"
    compile "org.eclipse.jetty:jetty-server:9.4.9.v20180320"
    compile "com.google.cloud.dataflow:google-cloud-dataflow-java-sdk-all:2.2.0"
    compile "org.slf4j:slf4j-api:1.7.5"
    compile "org.slf4j:slf4j-log4j12:1.7.5"
}
'''

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
