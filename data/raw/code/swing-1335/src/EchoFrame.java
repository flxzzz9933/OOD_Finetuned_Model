import java.awt.*;

import javax.swing.*;

public class EchoFrame extends JFrame {

  public EchoFrame() {
    super();

    //Manage the frame
    this.setSize(500, 150);
    this.setLocation(300, 500);
    this.setDefaultCloseOperation(EXIT_ON_CLOSE);
    this.setBackground(Color.MAGENTA);

    //Change the LayoutManager
    this.setLayout(new FlowLayout());

    //Construct your components
    JLabel echoLabel = new JLabel("Echo text here!");
    this.add(echoLabel);

    JTextField textInput = new JTextField(20);
    this.add(textInput);

    JButton echoButton = new JButton("Echo!");
    this.add(echoButton);

    //Compact the frame so everything just fits
    this.pack();
    this.setVisible(true);
  }
}
