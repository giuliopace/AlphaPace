package at.pwd.mancala;

import at.pwd.boardgame.game.base.WinState;
import at.pwd.boardgame.game.mancala.MancalaGame;
import at.pwd.boardgame.game.mancala.agent.MancalaAgent;
import at.pwd.boardgame.game.mancala.agent.MancalaAgentAction;

import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.apache.log4j.BasicConfigurator;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;


public class AlphaPace_backup implements MancalaAgent {
    private static final int STARTING_DEPTH = 3; //TODO: starting depth
    private static final int DEPTH_STEP = 2; //TODO: set step size for iterative deepening search
    private static int CURR_DEPTH;
    private static double CURR_BEST;
    private static List<String> BEST_STONE;

    @Override
    public MancalaAgentAction doTurn(int computationTime, MancalaGame mancalaGame) {
        //selfPlay(computationTime, mancalaGame);
        BasicConfigurator.configure();
        INDArray test = Nd4j.ones(1,14,6);

        try {
            String fullModel = new ClassPathResource("AlphaPace.h5").getFile().getPath();
            MultiLayerNetwork model = KerasModelImport.importKerasSequentialModelAndWeights(fullModel);
            double prediction = model.output(test).getDouble(0);
            System.out.println("Prediction: " + prediction);
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }

        return doTurn2(computationTime, mancalaGame);
    }

    public void selfPlay(int computationTime, MancalaGame mancalaGame) {
        boolean repeatTurn;
        boolean check;
        int i = 0;
        while (i < 1000) {
            i++;
            System.out.println("Game " + i + " in progress");
            MancalaGame simulation = new MancalaGame(mancalaGame);
            check = false;
            String player1 = "";
            String player2 = "";
            while (!check) {
                int currentPlayer = simulation.getState().getCurrentPlayer();
                String selectedStone = doTurn_sim(computationTime, simulation);
                repeatTurn = simulation.selectSlot(selectedStone);
                if (!repeatTurn) {
                    simulation.nextPlayer();
                }
                if (!(simulation.checkIfPlayerWins().getState() == WinState.States.NOBODY)) {
                    check = true;
                }
                ArrayList<Double> turn = new ArrayList<>();
                if (currentPlayer == 1) {
                    for (int j = 1; j <= 14; j++) {
                        double s = simulation.getState().stonesIn("" + j);
                        turn.add(s);
                    }
                    for (Double s : turn) {
                        player1 = player1 + s + " ";
                    }
                    player1 = player1 + "\n";
                    System.out.println("Player 1:\n" + player1);
                } else {
                    for (int j = 8; j <= 14; j++) {
                        double s = simulation.getState().stonesIn("" + j);
                        turn.add(s);
                    }
                    for (int j = 1; j <= 7; j++) {
                        double s = simulation.getState().stonesIn("" + j);
                        turn.add(s);
                    }
                    for (Double s : turn) {
                        player2 = player2 + s + " ";
                    }
                    player2 = player2 + "\n";
                    System.out.println("Player 2:\n" + player2);
                }
            }
            System.out.println(player1);
            try (PrintStream out = new PrintStream(new FileOutputStream("data/player1_" + i + ".txt"))) {
                out.print(player1);
                //out.flush();
                out.close();
            } catch (Exception e) {
                System.out.println("meeeeh");
            }
            try (PrintStream out = new PrintStream(new FileOutputStream("data/player2_" + i + ".txt"))) {
                out.print(player2);
                //out.flush();
                out.close();
            } catch (Exception e) {
                System.out.println("meeeeh");
            }
        }
    }

    public String doTurn_sim(int computationTime, MancalaGame mancalaGame) {
        long start_time = System.currentTimeMillis();
        MancalaGame simulation = new MancalaGame(mancalaGame);
        CURR_DEPTH = STARTING_DEPTH;
        CURR_BEST = Double.NEGATIVE_INFINITY;
        BEST_STONE = new ArrayList<String>();

        // System.out.println(start_time);
        long difference = System.currentTimeMillis() - start_time;

        // System.out.println("Player: " + simulation.getState().getCurrentPlayer());

        // for (int i = 1; i <= 14; i++) {
        //     System.out.println(simulation.getState().stonesIn("" + i));
        // }

        while ((difference) < (computationTime * 1000 - 500) && CURR_DEPTH < 11) { //TODO: change -1 to puffer of last move
            // System.out.println("AlphaBeta with Depth of " + CURR_DEPTH + ":");
            alphaBeta(simulation, CURR_DEPTH, Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY, true);
            CURR_DEPTH += DEPTH_STEP;
            difference = System.currentTimeMillis() - start_time;
            // System.out.println("Difference: " + difference);
        }

        // System.out.println(BEST_STONE);
        Random rand = new Random();
        String selectedStone = BEST_STONE.get(rand.nextInt(BEST_STONE.size()));
        // System.out.println(selectedStone);

        return selectedStone;
    }

    public MancalaAgentAction doTurn2(int computationTime, MancalaGame mancalaGame) {
        long start_time = System.currentTimeMillis();
        MancalaGame simulation = new MancalaGame(mancalaGame);
        CURR_DEPTH = STARTING_DEPTH;
        CURR_BEST = Double.NEGATIVE_INFINITY;
        BEST_STONE = new ArrayList<String>();

        System.out.println(start_time);
        long difference = System.currentTimeMillis() - start_time;

        System.out.println("Player: " + simulation.getState().getCurrentPlayer());

        for (int i = 1; i <= 14; i++) {
            System.out.println(simulation.getState().stonesIn("" + i));
        }

        while ((difference) < (computationTime * 1000 - 500) && CURR_DEPTH < 11) { //TODO: change -1 to puffer of last move
            // System.out.println("AlphaBeta with Depth of " + CURR_DEPTH + ":");
            alphaBeta(simulation, CURR_DEPTH, Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY, true);
            CURR_DEPTH += DEPTH_STEP;
            difference = System.currentTimeMillis() - start_time;
            System.out.println("Difference: " + difference);
        }

        System.out.println(BEST_STONE);
        Random rand = new Random();
        String selectedStone = BEST_STONE.get(rand.nextInt(BEST_STONE.size()));
        System.out.println(selectedStone);

        return new MancalaAgentAction(selectedStone);
    }

    private double evaluate(ArrayList<Integer> states) {
        return 0;
    }

    private double alphaBeta(MancalaGame simulation, int depth, double alpha, double beta, boolean maximizingPlayer) {
        if (depth == 0 || !(simulation.checkIfPlayerWins().getState() == WinState.States.NOBODY)) {
            if (simulation.getState().getCurrentPlayer() == 0) {
                return simulation.getState().stonesIn("8") - simulation.getState().stonesIn("1"); //TODO: heuristic of node
            } else {
                return simulation.getState().stonesIn("1") - simulation.getState().stonesIn("8");
            }
        }
        List<String> possibleMoves = simulation.getSelectableSlots();
        if (maximizingPlayer) {
            for (String m : possibleMoves) {
                MancalaGame nextSimulation = new MancalaGame(simulation);
                boolean repeatTurn = nextSimulation.selectSlot(m);
                if (!repeatTurn) {
                    nextSimulation.nextPlayer();
                    maximizingPlayer = !maximizingPlayer; //is this useful?
                    //we could just use repeat turn as maximizing player
                }
                alpha = Math.max(alpha, alphaBeta(nextSimulation, depth - 1, alpha, beta, maximizingPlayer));
                if (depth == CURR_DEPTH) {
                    if (alpha > CURR_BEST && !(BEST_STONE.contains(m))) {
                        CURR_BEST = alpha;
                        BEST_STONE = new ArrayList<String>();
                        BEST_STONE.add(m);
                        // System.out.println("Slot " + m + ": " + alpha);
                    } else if (alpha == CURR_BEST && !(BEST_STONE.contains(m))) {
                        BEST_STONE.add(m);
                    }
                }
                if (alpha >= beta) {
                    break; // beta cut-off
                }
            }
            return alpha;
        } else {
            for (String m : possibleMoves) {
                MancalaGame nextSimulation = new MancalaGame(simulation);
                boolean repeatTurn = nextSimulation.selectSlot(m);
                if (!repeatTurn) {
                    nextSimulation.nextPlayer();
                    maximizingPlayer = !maximizingPlayer; //is this useful?
                }
                beta = Math.min(beta, alphaBeta(nextSimulation, depth - 1, alpha, beta, maximizingPlayer));
                if (alpha >= beta) {
                    break; // alpha cut-off
                }
            }
            return beta;
        }
    }

    private double alphaBeta2(MancalaGame simulation, int depth, double alpha, double beta, boolean maximizingPlayer) {
        List<String> possibleMoves = simulation.getSelectableSlots();
        boolean repeatTurn;

        if (depth == 0 || !(simulation.checkIfPlayerWins().getState() == WinState.States.NOBODY)) {
            return simulation.getState().stonesIn("8") - simulation.getState().stonesIn("1"); //TODO: heuristic of node
        } else if (maximizingPlayer) {
            double value = Double.NEGATIVE_INFINITY;
            for (String m : possibleMoves) {
                MancalaGame nextSimulation = new MancalaGame(simulation);
                repeatTurn = nextSimulation.selectSlot(m);
                if (!repeatTurn) {
                    nextSimulation.nextPlayer();
                    maximizingPlayer = !maximizingPlayer;
                }
                value = Math.max(value, alphaBeta(nextSimulation, depth - 1, alpha, beta, maximizingPlayer));
                alpha = Math.max(alpha, value);
                if (depth == CURR_DEPTH) {
                    if (alpha > CURR_BEST && !(BEST_STONE.contains(m))) {
                        CURR_BEST = alpha;
                        BEST_STONE = new ArrayList<String>();
                        BEST_STONE.add(m);
                        // System.out.println("Slot " + m + ": " + alpha);
                    } else if (alpha == CURR_BEST && !(BEST_STONE.contains(m))) {
                        BEST_STONE.add(m);
                    }
                }
                if (alpha >= beta) {
                    break; // beta cut-off
                }
            }
            return value;
        } else {
            double value = Double.POSITIVE_INFINITY;
            for (String m : possibleMoves) {
                MancalaGame nextSimulation = new MancalaGame(simulation);
                repeatTurn = nextSimulation.selectSlot(m);
                if (!repeatTurn) {
                    nextSimulation.nextPlayer();
                    maximizingPlayer = !maximizingPlayer;
                }
                value = Math.min(value, alphaBeta(nextSimulation, depth - 1, alpha, beta, maximizingPlayer));
                beta = Math.min(beta, value);
                if (alpha >= beta) {
                    break; // alpha cut-off
                }
            }
            return value;
        }
    }

    @Override
    public String toString() {
        return "AlphaPace Agent";
    }
}

/*
function alphabeta(node, depth, α, β, maximizingPlayer) is
    if depth = 0 or node is a terminal node then
        return the heuristic value of node
    if maximizingPlayer then
        value := −∞
        for each child of node do
            value := max(value, alphabeta(child, depth − 1, α, β, FALSE))
            α := max(α, value)
            if α ≥ β then
                break (* β cut-off *)
        return value
    else
        value := +∞
        for each child of node do
            value := min(value, alphabeta(child, depth − 1, α, β, TRUE))
            β := min(β, value)
            if α ≥ β then
                break (* α cut-off *)
        return value
(* Initial call *)
alphabeta(origin, depth, −∞, +∞, TRUE)
 */