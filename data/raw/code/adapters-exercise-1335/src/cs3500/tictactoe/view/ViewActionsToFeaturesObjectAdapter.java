package cs3500.tictactoe.view;

import java.util.Scanner;

import cs3500.tictactoe.provider.view.Features;

public class ViewActionsToFeaturesObjectAdapter implements Features  {

  private ViewActions adaptee;

  public ViewActionsToFeaturesObjectAdapter(ViewActions adaptee) {
    this.adaptee = adaptee;
  }

  @Override
  public void processCommand(String input) {
    Scanner scan = new Scanner(input);
    int row = scan.nextInt();
    int col = scan.nextInt();
    this.adaptee.handleCellClick(row -1, col -1);
  }
}
