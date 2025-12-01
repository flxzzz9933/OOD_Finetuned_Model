import java.awt.Color;
import java.awt.FlowLayout;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JTextField;

public class EchoFrame extends JFrame {

  public EchoFrame() {
    super();
    this.setSize(700, 300);
    this.setLocation(600, 500);

    //To put the frame in the middle of the screen, use
    //this.setLocationRelativeTo(null);

    this.setDefaultCloseOperation(EXIT_ON_CLOSE);
    this.setBackground(Color.MAGENTA);

    //Set the manager so everything is added in a line if possible
    this.setLayout(new FlowLayout());
    this.add(new JLabel("Echo text!"));
    this.add(new JTextField(20));
    this.add(new JButton("Echo!"));

    this.pack();
    this.setVisible(true);
  }
}
