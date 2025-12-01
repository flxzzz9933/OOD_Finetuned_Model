package cs3500.tictactoe;

import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;

import javax.swing.*;

public class TTTFrame extends JFrame implements TTTView {

  private TTTPanel panel;

  public TTTFrame(ReadonlyTTTModel model) {
    super();
    setSize(900, 900);
    setDefaultCloseOperation(EXIT_ON_CLOSE);
    panel = new TTTPanel(model);
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
    this.addKeyListener(new KeyListener() {
      @Override
      public void keyTyped(KeyEvent e) {
        if(e.getKeyChar() == 'q') {
          observer.quit();
        }
      }

      @Override
      public void keyPressed(KeyEvent e) {

      }

      @Override
      public void keyReleased(KeyEvent e) {

      }
    });
  }
}
