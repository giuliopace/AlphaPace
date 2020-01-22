package at.pwd.mancala;

import at.pwd.boardgame.game.base.WinState;
import at.pwd.boardgame.game.mancala.MancalaGame;
import at.pwd.boardgame.game.mancala.agent.MancalaAgent;
import at.pwd.boardgame.game.mancala.agent.MancalaAgentAction;

import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;


public class AlphaPace_should_be implements MancalaAgent {
    private static final int DEPTH = 10;
    private int currentPlayer;
    private String currentBest;

    @Override
    public MancalaAgentAction doTurn(int computationTime, MancalaGame mancalaGame) {
        selfPlay(computationTime, mancalaGame);
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

    public MancalaAgentAction doTurn2(int computationTime, MancalaGame initialGame) {
        currentPlayer = initialGame.getState().getCurrentPlayer();
        currentBest = null;

        alphabeta(initialGame, DEPTH, Integer.MIN_VALUE, Integer.MAX_VALUE, true);

        return new MancalaAgentAction(currentBest);
    }

    public String doTurn_sim(int computationTime, MancalaGame mancalaGame) {
        currentPlayer = mancalaGame.getState().getCurrentPlayer();
        currentBest = null;

        alphabeta(mancalaGame, DEPTH, Integer.MIN_VALUE, Integer.MAX_VALUE, true);

        return currentBest;
    }

    private int heuristic(MancalaGame node) {
        String ownDepot = node.getBoard().getDepotOfPlayer(currentPlayer);
        String enemyDepot = node.getBoard().getDepotOfPlayer(1 - currentPlayer);
        return node.getState().stonesIn(ownDepot) - node.getState().stonesIn(enemyDepot);
    }

    private int alphabeta(MancalaGame node, int depth, int alpha, int beta, boolean maximizingPlayer) {
        if (depth == 0 || node.checkIfPlayerWins().getState() != WinState.States.NOBODY) {
            return heuristic(node);
        }

        List<String> legalMoves = node.getSelectableSlots();
        for (String move : legalMoves) {
            MancalaGame newGame = new MancalaGame(node);
            boolean moveAgain = newGame.selectSlot(move);
            if (!moveAgain) {
                newGame.nextPlayer();
            }

            if (maximizingPlayer) {
                int oldAlpha = alpha;
                alpha = Math.max(alpha, alphabeta(newGame, depth - 1, alpha, beta, moveAgain));
                if (depth == DEPTH && (oldAlpha < alpha || currentBest == null)) {
                    currentBest = move;
                }
            } else {
                beta = Math.min(beta, alphabeta(newGame, depth - 1, alpha, beta, !moveAgain));
            }

            if (beta <= alpha) {
                break;
            }
        }
        return maximizingPlayer ? alpha : beta;
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