package cs3500.tictactoe.view;

import cs3500.tictactoe.model.TicTacToe;
import cs3500.tictactoe.provider.view.TicTacToeView;

public class TicTacToeViewToTTTViewAdapter implements TTTView {

  private TicTacToeView adaptee;
  private TicTacToe model;

  public TicTacToeViewToTTTViewAdapter(TicTacToe model, TicTacToeView adaptee) {
    if(adaptee == null || model == null) {
      throw new IllegalArgumentException("Can't be null");
    }
    this.adaptee = adaptee;
    this.model = model;
  }

  @Override
  public void addClickListener(ViewActions listener) {
    this.adaptee.addFeatures(new ViewActionsToFeaturesAdapter(listener));
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
