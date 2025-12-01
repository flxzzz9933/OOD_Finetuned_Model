package cs3500.tictactoe.controller;

import cs3500.tictactoe.model.TicTacToe;
import cs3500.tictactoe.view.TTTView;
import cs3500.tictactoe.view.ViewActions;

public class TicTacToeGUIController implements TicTacToeController, ViewActions {

  private final TTTView view;
  private TicTacToe model;
  public TicTacToeGUIController(TTTView view) {
    this.view = view;
  }

  @Override
  public void playGame(TicTacToe m) {
    this.model = m;
    this.view.addClickListener(this);
    this.view.makeVisible();
  }

  @Override
  public void handleCellClick(int row, int col) {
    try {
      model.move(row, col);
    } catch (IllegalArgumentException ex) {
      //tell the user somehow the arguments were wrong
    } catch (IllegalStateException ex) {
      //tell the user somehow they made a mistake choosing a cell
      //or the game is over
    }
    view.refresh();
  }
}
