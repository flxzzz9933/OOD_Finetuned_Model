package cs3500.tictactoe;

public class TTTGUIController implements TicTacToeController, ViewActions {

  private TicTacToe model;
  private TTTView view;

  private int row, col;

  public TTTGUIController(TTTView view) {
    //Check if view is null and throw IAE if so
    this.view = view;
  }

  @Override
  public void playGame(TicTacToe m) {
    //check if model is null and throw IAE if so
    this.model = m;
    this.view.subscribe(this);
    this.view.makeVisible();
  }

  @Override
  public void click(int row, int col) {
    this.row = row;
    this.col = col;
  }

  @Override
  public void confirm() {
    model.move(row, col);
    row = -1;
    col = -1;
    view.refresh();
  }
}
