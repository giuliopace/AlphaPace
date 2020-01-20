package at.pwd.mancala;

import at.pwd.boardgame.game.base.WinState;
import at.pwd.boardgame.game.mancala.MancalaGame;
import at.pwd.boardgame.game.mancala.MancalaState;
import at.pwd.boardgame.game.mancala.agent.MancalaAgent;
import at.pwd.boardgame.game.mancala.agent.MancalaAgentAction;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;


public class AlphaPace implements MancalaAgent {
    private static final int STARTING_DEPTH = 3; //TODO: starting depth
    private static final int DEPTH_STEP = 2; //TODO: set step size for iterative deepening search
    private static int CURR_DEPTH;
    private static double CURR_BEST;
    private static List<String> BEST_STONE;

    @Override
    public MancalaAgentAction doTurn(int computationTime, MancalaGame mancalaGame) {
        long start_time = System.currentTimeMillis();
        MancalaGame simulation = new MancalaGame(mancalaGame);
        CURR_DEPTH = STARTING_DEPTH;
        CURR_BEST = Double.NEGATIVE_INFINITY;
        BEST_STONE = new ArrayList<String>();

        System.out.println(start_time);
        long difference = System.currentTimeMillis() - start_time;

        while ((difference) < (computationTime * 1000 - 500) && CURR_DEPTH < 15) { //TODO: change -1 to puffer of last move
            // System.out.println("AlphaBeta with Depth of " + CURR_DEPTH + ":");
            alphaBeta(simulation, CURR_DEPTH, Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY, true);
            CURR_DEPTH += DEPTH_STEP;
            difference = System.currentTimeMillis() - start_time;
            System.out.println(difference);
        }

        System.out.println(BEST_STONE);
        Random rand = new Random();
        String selectedStone = BEST_STONE.get(rand.nextInt(BEST_STONE.size()));
        System.out.println(selectedStone);

        return new MancalaAgentAction(selectedStone);
    }

    private double alphaBeta(MancalaGame simulation, int depth, double alpha, double beta, boolean maximizingPlayer) {
        List<String> possibleMoves = simulation.getSelectableSlots();
        boolean repeatTurn;

        if (depth == 0 || !(simulation.checkIfPlayerWins().getState() == WinState.States.NOBODY)) {
            return simulation.getState().stonesIn("1") - simulation.getState().stonesIn("8"); //TODO: heuristic of node
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