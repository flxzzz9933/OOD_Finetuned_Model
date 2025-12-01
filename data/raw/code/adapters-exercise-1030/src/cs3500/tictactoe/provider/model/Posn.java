package cs3500.tictactoe.provider.model;

public class Posn {

  private final int row;
  private final int col;

  public Posn(int row, int col) {
    this.row = row;
    this.col = col;
  }

  public int row() {
    return row;
  }

  public int col() {
    return col;
  }
}

