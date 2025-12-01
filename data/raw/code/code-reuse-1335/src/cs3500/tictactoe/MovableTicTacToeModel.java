package cs3500.tictactoe;

import java.util.ArrayDeque;
import java.util.Queue;

public class MovableTicTacToeModel extends TicTacToeModel {

  //Any new state?

  //Queue<Integer> // each pair is one coordinate

  //int[][] board // for each coordinate, how long has a piece been there

  //int numTurns // how many turns have passed so we don't remove too early

  class Posn {
    int row;
    int col;

    Posn(int row, int col) {
      this.row = row;
      this.col = col;
    }
  }

  private Queue<Posn> moves;

  public MovableTicTacToeModel() {
    super();
    this.moves = new ArrayDeque<>();
  }

  @Override
  public void move(int row, int col) {
    super.move(row, col);

    if(moves.size() == 6) {
      Posn oldest = moves.remove();
      board[oldest.row][oldest.col] = null;
    }

    moves.add(new Posn(row, col));
  }

}
