package cs3500.tictactoe.provider.view;

import cs3500.tictactoe.model.Player;

public interface TicTacToeView {

  void addFeatures(Features feat);

  void refresh();

  void makeVisible();

  void updateBoard(Player[][] board);
}
