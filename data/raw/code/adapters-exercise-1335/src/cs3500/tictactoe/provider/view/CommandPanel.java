package cs3500.tictactoe.provider.view;

import javax.swing.*;

public class CommandPanel extends JPanel {

  private final JTextField field;
  private final JButton button;
  public CommandPanel() {
    this.field = new JTextField(20);
    this.add(field);
    this.button = new JButton("Execute");
    this.add(button);
  }

  public void addFeatures(Features feat) {
    this.button.addActionListener((evt) -> feat.processCommand(this.field.getText()));
  }

  public void clearText() {
    this.field.setText("");
  }
}
