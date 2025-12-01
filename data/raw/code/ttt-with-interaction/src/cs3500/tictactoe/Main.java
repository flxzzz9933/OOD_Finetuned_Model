package cs3500.tictactoe;

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
    TTTView view = new TTTFrame(model);
    // 3. Create an instance of the controller, passing the view to its constructor.
    TicTacToeController controller =
        new TicTacToeGUIController(view);
    // 4. Call playGame() on the controller.
    controller.playGame(model);
  }
}
