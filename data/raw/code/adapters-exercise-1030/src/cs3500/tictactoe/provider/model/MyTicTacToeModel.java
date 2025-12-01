package cs3500.tictactoe.provider.model;

public interface MyTicTacToeModel {

  void move(Posn pos);

  String getMarkAt(Posn pos);

  boolean isGameOver();

  String winner();

  int getWidth();

  int getHeight();
}
