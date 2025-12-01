package cs3500.tictactoe;


import java.util.ArrayDeque;
import java.util.List;
import java.util.Queue;

public class MovableTicTacToeModel extends TicTacToeModel {

  //counter: number of plays that have taken place
  //List<Integer>: every 2 integers is a location
  //int[][]: each row is a coordinate
  //Queue<Point>

  class Posn {
    int x;
    int y;

    Posn(int x, int y) {
      this.x = x;
      this.y = y;
    }
  }

  private int counter;
  private Queue<Posn> positions;

  public MovableTicTacToeModel() {
    super();
    counter = 0;
    positions = new ArrayDeque<>();
  }

  @Override
  public void move(int row, int col) {
    super.move(row, col);
    //if there are six pieces already on the board
    if(positions.size() == 6) { //or check counter == 6
      //remove the oldest one
      Posn oldest = positions.remove();
      board[oldest.x][oldest.y] = null;
      //counter--
    }
    positions.add(new Posn(row, col));
    //counter++;
  }

}
