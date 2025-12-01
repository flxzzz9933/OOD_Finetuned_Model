import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.*;

public class EchoPanel extends JPanel {

  private final JTextField textInput;
  private final JLabel echoLabel;

  private final IModel model;

  public EchoPanel(IModel model) {
    super();

    this.model = model;

    //By default, panels have a FlowLayout

    echoLabel = new JLabel("Echo text here!");
    this.add(echoLabel);

    textInput = new JTextField(20);
    this.add(textInput);

    JButton echoButton = new JButton("Echo!");
    echoButton.setActionCommand("echo");
    echoButton.addActionListener(this);
    this.add(echoButton);

    JButton exitButton = new JButton("Exit");
    exitButton.setActionCommand("exit");
    exitButton.addActionListener(this);
    this.add(exitButton);
  }


    //JOptionPane.showMessageDialog(this, "Hurray");
    /*
    JFrame frame = new JFrame("Hurray!");
    frame.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
    frame.setVisible(true);
     */
  }
}
