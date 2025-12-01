package view;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.*;

import model.IModel;

public class EchoPanel extends JPanel {

  private final JTextField textInput;
  private final JLabel echoLabel;
  private JButton echoButton, exitButton;

  //private final IModel model; //Used to be needed, but no longer as there is no listener here.

  public EchoPanel(IModel model) {
    super();

    //this.model = model;

    //By default, panels have a FlowLayout

    echoLabel = new JLabel("Echo text here!");
    this.add(echoLabel);

    textInput = new JTextField(20);
    this.add(textInput);

    echoButton = new JButton("Echo!");
    echoButton.setActionCommand("echo");
    //echoButton.addActionListener(this); //This isn't a listener. So who do we pass in? And where?
    this.add(echoButton);

    exitButton = new JButton("Exit");
    exitButton.setActionCommand("exit");
    //exitButton.addActionListener(this); //This isn't a listener. So who do we pass in? And where?
    this.add(exitButton);
  }

  public String getInput() {
    return textInput.getText();
  }

  public void displayText(String text) {
    echoLabel.setText(text);
  }

  public void clearInput() {
    textInput.setText("");
  }

  public void setViewActions(ViewActions actions) {
    echoButton.addActionListener(new EchoActionListener(actions));
    exitButton.addActionListener(
        (ActionEvent evt) -> { actions.exitProgram();}
    );
  }

  class EchoActionListener implements ActionListener {

    private ViewActions actions;

    public EchoActionListener(ViewActions actions) {
      this.actions = actions;
    }

    @Override
    public void actionPerformed(ActionEvent e) {
      actions.displayText(textInput.getText());
    }
  }


}
