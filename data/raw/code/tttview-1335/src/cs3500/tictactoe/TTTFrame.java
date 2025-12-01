package cs3500.tictactoe;

import javax.swing.JFrame;

public class TTTFrame extends JFrame implements TTTView {

  public TTTFrame(ReadonlyTTTModel model) {
    super();
    this.setSize(900, 900);
    this.setDefaultCloseOperation(EXIT_ON_CLOSE);
    TTTPanel panel = new TTTPanel(model);
    this.add(panel);
  }

  @Override
  public void refresh() {
    this.repaint();
  }

  @Override
  public void makeVisible() {
    this.setVisible(true);
  }

  @Override
  public void subscribe(ViewActions observer) {
    this.panel.subscribe(observer);
    //TODO: Add KeyListener here
  }
}
