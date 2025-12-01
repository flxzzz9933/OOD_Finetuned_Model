package cs3500.tictactoe.view;

import cs3500.tictactoe.model.TicTacToe;
import cs3500.tictactoe.provider.view.TicTacToeView;

public class TicTacToeViewToTTTViewAdapter implements TTTView {

  private TicTacToeView adaptee;
  private TicTacToe model;

  public TicTacToeViewToTTTViewAdapter(TicTacToeView adaptee, TicTacToe model) {
    if(adaptee == null) {
      throw new IllegalArgumentException("Cannot be null");
    }
    this.adaptee = adaptee;
    this.model = model;
  }

  // processCommands is 1-index based row, then col
  // handleCellClick is 0-index based row and col
  @Override
  public void addClickListener(ViewActions listener) {
    //TODO: Find a way to make sure the class adapter has the model itself
    // since it IS the controller now
    //this.adaptee.addFeatures(new ViewActionsToFeaturesAdapter(this));

    this.adaptee.addFeatures(new ViewActionsToFeaturesObjectAdapter(listener));
  }

  @Override
  public void refresh() {
    this.adaptee.updateBoard(model.getBoard());
    this.adaptee.refresh();
  }

  @Override
  public void makeVisible() {
    this.adaptee.makeVisible();
  }
}
