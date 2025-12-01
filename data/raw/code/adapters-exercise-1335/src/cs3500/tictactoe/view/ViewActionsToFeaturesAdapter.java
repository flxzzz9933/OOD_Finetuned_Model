package cs3500.tictactoe.view;

import java.util.Scanner;

import cs3500.tictactoe.controller.TicTacToeGUIController;
import cs3500.tictactoe.provider.view.Features;

public class ViewActionsToFeaturesAdapter extends TicTacToeGUIController
    implements Features {

  public ViewActionsToFeaturesAdapter(TTTView view) {
    super(view);
  }

  @Override
  public void processCommand(String input) {
    Scanner scan = new Scanner(input);
    int row = scan.nextInt();
    int col = scan.nextInt();
    super.handleCellClick(row-1, col-1);
  }
}
