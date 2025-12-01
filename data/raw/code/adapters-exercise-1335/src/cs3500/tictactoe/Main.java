package cs3500.tictactoe;

import cs3500.tictactoe.controller.TicTacToeController;
import cs3500.tictactoe.controller.TicTacToeGUIController;
import cs3500.tictactoe.model.TicTacToe;
import cs3500.tictactoe.model.TicTacToeModel;
import cs3500.tictactoe.provider.view.TicTacToeFrame;
import cs3500.tictactoe.view.TTTFrame;
import cs3500.tictactoe.view.TTTView;
import cs3500.tictactoe.view.TicTacToeViewToTTTViewAdapter;

/**
 * Run a Tic Tac Toe game interactively.
 */
public class Main {
  /**
   * Run a Tic Tac Toe game interactively.
   */
  public static void main(String[] args) {
    // Old News: console-based game:
    //new TicTacToeConsoleController(new InputStreamReader(System.in),
    //    System.out).playGame(new TicTacToeModel());

    // New Hotness: Graphical User Interface:
    // 1. Create an instance of the model.
    TicTacToe model = new TicTacToeModel();
    // 2. Create an instance of the view.
    TTTView view = new TicTacToeViewToTTTViewAdapter(new TicTacToeFrame(),
        model);
    // 3. Create an instance of the controller, passing the view to its constructor.
    TicTacToeController controller =
        new TicTacToeGUIController(view);
    // 4. Call playGame() on the controller.
    controller.playGame(model);
  }
}
