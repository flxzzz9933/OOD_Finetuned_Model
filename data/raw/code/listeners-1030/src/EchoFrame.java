import java.awt.Color;
import java.awt.FlowLayout;
import java.awt.event.ActionListener;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JTextField;

public class EchoFrame extends JFrame implements GUIView {

  private final EchoPanel echoPanel;
  public EchoFrame(IModel model) {
    super();
    this.setSize(700, 300);
    this.setLocation(600, 500);

    //To put the frame in the middle of the screen, use
    //this.setLocationRelativeTo(null);

    this.setDefaultCloseOperation(EXIT_ON_CLOSE);
    this.setBackground(Color.MAGENTA);

    //Set the manager so everything is added in a line if possible
    this.setLayout(new FlowLayout());

    echoPanel = new EchoPanel(model);
    this.add(echoPanel);

    this.pack();
    this.setVisible(true);
  }

  @Override
  public String getInput() {
    return echoPanel.getInput();
  }

  @Override
  public void displayText(String string) {
    echoPanel.displayText(string);
  }

  @Override
  public void clearInput() {
    echoPanel.clearInput();
  }

}
