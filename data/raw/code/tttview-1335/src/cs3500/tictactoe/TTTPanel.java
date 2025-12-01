package cs3500.tictactoe;

import java.awt.*;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.geom.AffineTransform;
import java.awt.geom.Point2D;

import javax.swing.JPanel;

public class TTTPanel extends JPanel {

  private final ReadonlyTTTModel model;

  public TTTPanel(ReadonlyTTTModel model) {
    //Should check if it's null and if so, throw an IAE
    this.model = model;
  }

  @Override
  protected void paintComponent(Graphics g) {
    super.paintComponent(g);
    Graphics2D g2d = (Graphics2D)g.create();
    //Now we can draw stuff

    g2d.transform(getLogicalToPhysicalXForm());

    int width = this.getWidth();
    int height = this.getHeight();

    drawLine(g2d, 1, 0, 1, 3);
    drawLine(g2d, 2, 0, 2, 3);
    drawLine(g2d, 0, 1, 3, 1);
    drawLine(g2d, 0, 2, 3, 2);

    //g2d.setColor(Color.RED);
    //g2d.drawLine(0,0, width, height);

    //drawMarks(g2d);
  }

  private void drawLine(Graphics2D g2d, int srcRow, int srcCol,
                        int dstRow, int dstCol) {
    AffineTransform modelToLogical = getModelToLogicalXForm();
    Point2D src = modelToLogical.transform(new Point(srcCol, srcRow), null);
    Point2D dst = modelToLogical.transform(new Point(dstCol, dstRow), null);

    g2d.drawLine((int)src.getX(),
        (int)src.getY(),
        (int)dst.getX(),
        (int)dst.getY());
  }

  private Dimension getLogicalDimensions() {
    return new Dimension(30, 30);
  }

  private AffineTransform getLogicalToPhysicalXForm() {
    AffineTransform xform = new AffineTransform();
    xform.scale(this.getWidth() / getLogicalDimensions().getWidth(),
        this.getHeight() / getLogicalDimensions().getHeight());
    return xform;
  }

  private AffineTransform getModelToLogicalXForm() {
    AffineTransform xform = new AffineTransform();
    xform.scale(getLogicalDimensions().getWidth() / 3,
        getLogicalDimensions().getHeight() / 3);
    return xform;
  }

  private AffineTransform getModelToPhysicalXForm() {
    AffineTransform xform = new AffineTransform();
    xform.scale(this.getWidth()/3, this.getHeight()/3);
    return xform;
  }

  private void drawMarks(Graphics2D g2d) {
    for(int row = 0; row < 3; row++) {
      for(int col = 0; col < 3; col++) {
        int xPos = (col * this.getWidth())/3;
        int yPos = (row * this.getHeight())/3;
        g2d.drawOval(xPos, yPos, 30, 30);
      }
    }
  }

  public void subscribe(ViewActions observer) {
    this.addMouseListener(new MouseListener() {
      @Override
      public void mouseClicked(MouseEvent e) {
        AffineTransform physicalToLogical = getLogicalToPhysicalXForm();
        physicalToLogical.invert();
      }

      @Override
      public void mousePressed(MouseEvent e) {

      }

      @Override
      public void mouseReleased(MouseEvent e) {

      }

      @Override
      public void mouseEntered(MouseEvent e) {

      }

      @Override
      public void mouseExited(MouseEvent e) {

      }
    });
  }
}
