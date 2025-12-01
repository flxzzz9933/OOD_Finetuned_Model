package cs3500.tictactoe.view;

import cs3500.tictactoe.provider.view.Features;

public class ViewActionsToFeaturesAdapter implements Features {

  //input is 1-index based coordinates.
  //recall ViewActions uses 0-index based coordinates
  private ViewActions adaptee;

  public ViewActionsToFeaturesAdapter(ViewActions adaptee) {
    this.adaptee = adaptee;
  }

  @Override
  public void processCommand(String input) {
    //parse the input to get a row and col
    String[] indices = input.split(" "); //index 0 is row, index 1 is col
    //pass that into handleCellClick
    this.adaptee.handleCellClick(
        Integer.parseInt(indices[0])-1,
        Integer.parseInt(indices[1])-1);
  }
}
