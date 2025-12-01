package cs3500.tictactoe.provider.view;

import java.awt.*;

import javax.swing.*;

import cs3500.tictactoe.model.Player;

public class TicTacToeFrame extends JFrame implements TicTacToeView {

  private final CommandPanel commandPanel;
  private final DisplayPanel displayPanel;

  public TicTacToeFrame() {
    super();
    setSize(800, 800);
    setDefaultCloseOperation(EXIT_ON_CLOSE);
    this.setLayout(new BorderLayout());

    this.displayPanel = new DisplayPanel();
    this.add(this.displayPanel, BorderLayout.CENTER);

    this.commandPanel = new CommandPanel();
    this.add(this.commandPanel, BorderLayout.PAGE_END);
  }

  @Override
  public void addFeatures(Features feat) {
    this.commandPanel.addFeatures(feat);
  }

  @Override
  public void refresh() {
    this.commandPanel.clearText();
    this.repaint();
  }

  @Override
  public void makeVisible() {
    this.setVisible(true);
  }

  @Override
  public void updateBoard(Player[][] board) {
    this.displayPanel.updateBoard(board);
  }
}
