package cs3500.turtle.view;

import cs3500.turtle.model.Position2D;
import cs3500.turtle.tracingmodel.Line;

import javax.swing.JPanel;
import java.awt.*;
import java.awt.geom.AffineTransform;
import java.util.ArrayList;
import java.util.List;

public class TurtlePanel extends JPanel {
  List<Line> curLines;
  double headingDegrees;
  Position2D pos;

  TurtlePanel() {
    this.curLines = new ArrayList<>();
    this.pos = new Position2D(-1, -1);
  }
  @Override
  protected void paintComponent(Graphics g) {
    super.paintComponent(g);

    // TODO: Fill this in to draw the lines
    // Note: generally a good idea to copy the existing graphics
    // so that if you change its state (color, transform, etc)
    // you won't modify the original `g` that was passed in.
    Graphics2D g2d = (Graphics2D) g.create();

    g2d.drawOval(30, 30, 30, 30);
    g2d.fillRect(300, 350, 100, 50);


    // Use methods on the Graphics2D class to drawOval or fillOval,
    // drawLine, getColor, setColor, getTransform,
    // translate, scale, rotate, etc...
  }

}
