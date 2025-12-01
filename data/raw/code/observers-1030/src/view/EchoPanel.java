package view;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.*;

public class EchoPanel extends JPanel {

  private JLabel echoLabel;
  private JTextField inputField;

  private final JButton echoButton;
  private final JButton exitButton;

  public EchoPanel() {
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

  public void setViewActions(ViewActions actions) {
    echoButton.addActionListener(new EchoActionListener(actions));

    //ActionListeners are function objects, so we can replace them with lambdas
    exitButton.addActionListener( (evt) -> { actions.exitProgram(); } );
  }

  class EchoActionListener implements ActionListener {

    private ViewActions actions;

    public EchoActionListener(ViewActions actions) {
      this.actions = actions;
    }

    @Override
    public void actionPerformed(ActionEvent e) {
      actions.displayText(EchoPanel.this.getInput());
      //If we want access to the field itself
      //actions.displayText(EchoPanel.this.textInput.getText());
    }
  }
}

