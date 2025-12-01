package cs3500.tictactoe;

public class TTTGUIController implements TicTacToeController, ViewActions {

  private TicTacToe model;
  private TTTView view;

  public TTTGUIController(TTTView view) {
    if (view == null) {
      throw new IllegalArgumentException("Bad view");
    }
    this.view = view;
  }

  @Override
  public void playGame(TicTacToe m) {
    this.model = m;
    this.view.subscribe(this);
    this.view.makeVisible();
  }

  @Override
  public void placeMark(int row, int col) {
    model.move(row, col);
    view.refresh();
  }

  @Override
  public void quit() {
    System.exit(0);
  }
}
