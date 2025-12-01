package cs3500.tictactoe.provider;

import cs3500.tictactoe.model.TicTacToe;
import cs3500.tictactoe.model.TicTacToeModel;
import cs3500.tictactoe.provider.view.TicTacToeFrame;
import cs3500.tictactoe.provider.view.TicTacToeView;

public class Main {

  public static void main(String[] args) {
    TicTacToe model = new TicTacToeModel();
    TicTacToeView view = new TicTacToeFrame();
    view.makeVisible();
  }

}
