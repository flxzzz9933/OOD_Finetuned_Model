package cs3500.tictactoe.view;

import javax.swing.*;

import cs3500.tictactoe.model.ReadonlyTTTModel;

public class TTTFrame extends JFrame implements TTTView {

  private TTTPanel panel;

  public TTTFrame(ReadonlyTTTModel model) {
    super();
    setSize(800, 800);
    setDefaultCloseOperation(EXIT_ON_CLOSE);
    panel = new TTTPanel(model);
    this.add(panel);
  }

  @Override
  public void addClickListener(ViewActions observer) {
    panel.addClickListener(observer);
  }

  @Override
  public void refresh() {
    this.repaint();
  }

  @Override
  public void makeVisible() {
    setVisible(true);
  }
}
