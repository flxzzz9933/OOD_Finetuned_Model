import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.*;

public class EchoPanel extends JPanel {

  private JLabel echoLabel;
  private JTextField inputField;

  private final JButton echoButton;
  private final JButton exitButton;

  public EchoPanel(IModel model) {
    super();

    this.echoLabel = new JLabel("Echo text!");
    this.add(echoLabel);

    this.inputField = new JTextField(20);
    this.add(inputField);

    echoButton = new JButton("Echo!");
    echoButton.setActionCommand("echo");
    //echoButton.addActionListener(this); //This no longer works! So how do we add the listener?! (Observer pattern)
    this.add(echoButton);

    exitButton = new JButton("Exit");
    exitButton.setActionCommand("exit");
    //exitButton.addActionListener(this); //This no longer works! So how do we add the listener?! (Observer pattern)
    this.add(exitButton);
  }

  public String getInput() {
    return inputField.getText();
  }

  public void displayText(String string) {
    echoLabel.setText(string);
  }

  public void clearInput() {
    inputField.setText("");
  }
}

