package view;

import java.awt.*;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;

import javax.swing.*;

import model.IModel;

public class EchoFrame extends JFrame implements IGUIView {

  private EchoPanel echoPanel;
  public EchoFrame(IModel model) {
    super();

    //Manage the frame
    this.setSize(500, 150);
    this.setLocation(300, 500);
    this.setDefaultCloseOperation(EXIT_ON_CLOSE);
    this.setBackground(Color.MAGENTA);

    //Change the LayoutManager
    this.setLayout(new FlowLayout());

    //Construct your components
    this.echoPanel = new EchoPanel(model);
    this.add(echoPanel);

    //Compact the frame so everything just fits
    this.pack();
    this.setVisible(true);
  }

  @Override
  public String getInput() {
    return echoPanel.getInput();
  }

  @Override
  public void displayText(String text) {
    echoPanel.displayText(text);
  }

  @Override
  public void clearInput() {
    echoPanel.clearInput();
  }

  @Override
  public void setViewActions(ViewActions actions) {
    echoPanel.setViewActions(actions);
    this.addKeyListener(new KeyListener() {
      @Override
      public void keyTyped(KeyEvent e) {
        if(e.getKeyChar() == 'x') {
          actions.exitProgram();
        }
      }

      @Override
      public void keyPressed(KeyEvent e) {
        //doing nothing on purpose
      }

      @Override
      public void keyReleased(KeyEvent e) {
        //doing nothing on purpose
      }
    });
  }

  @Override
  public void makeVisible() {
    this.setVisible(true);
  }

  @Override
  public void refresh() {
    this.repaint();
    this.requestFocus();
  }
}
