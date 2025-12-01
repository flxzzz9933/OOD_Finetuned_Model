package cs3500.tictactoe;

import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.geom.AffineTransform;
import java.awt.geom.NoninvertibleTransformException;
import java.awt.geom.Point2D;

import javax.swing.*;

public class TTTPanel extends JPanel {

  private final ReadonlyTTTModel model;

  public TTTPanel(ReadonlyTTTModel model) {
    super();
    //Check if it's null and throw an IAE
    this.model = model;
  }

  @Override
  protected void paintComponent(Graphics g) {
    super.paintComponent(g);
    Graphics2D g2d = (Graphics2D)g.create();

    g2d.setColor(Color.BLACK);
    int height = this.getHeight();
    int width = this.getWidth();

    g2d.transform(getXformForLogicalToPhysical());

    drawLine(g2d, 1, 0, 1, 3);
    drawLine(g2d, 2, 0, 2, 3);
    drawLine(g2d, 0, 1, 3, 1);
    drawLine(g2d, 0, 2, 3, 2);

    drawBoard(g2d);
  }

  private void drawLine(Graphics2D g2d, int row, int col, int endRow, int endCol) {
    AffineTransform modelToLogical = getXFormForModelToLogical();
    Point2D src = modelToLogical.transform(new Point(col, row), null);
    Point2D dst = modelToLogical.transform(new Point(endCol, endRow), null);

    g2d.drawLine((int)src.getX(),
        (int)src.getY(),
        (int)dst.getX(),
        (int)dst.getY());
  }

  //Works, but lines are too big and look ugly. Let's not use.
  private AffineTransform getXFormForModelToPhysical() {
    AffineTransform xform = new AffineTransform();
    xform.scale(getWidth() / 3, getHeight() / 3);
    return xform;
  }

  private Dimension getLogicalDimensions() {
    return new Dimension(30, 30);
  }

  private AffineTransform getXFormForModelToLogical() {
    AffineTransform xform = new AffineTransform();
    xform.scale(getLogicalDimensions().getWidth() / 3,
        getLogicalDimensions().getHeight() / 3);
    return xform;
  }

  private AffineTransform getXformForLogicalToPhysical() {
    AffineTransform xform = new AffineTransform();
    xform.scale(this.getWidth() / getLogicalDimensions().getWidth(),
        this.getHeight() / getLogicalDimensions().getHeight());
    return xform;
  }

  private void drawOval(Graphics2D g2d, int row, int col, int width, int height) {
    AffineTransform modelToLogical = getXFormForModelToLogical();
    Point2D src = modelToLogical.transform(new Point(col, row), null);
    Point2D dst = modelToLogical.transform(new Point(width, height), null);

    g2d.drawOval((int)src.getX(),
        (int)src.getY(),
        (int)dst.getX(),
        (int)dst.getY());
  }

  private void drawBoard(Graphics2D g2d) {
    for(int row = 0; row < 3; row++) {
      for(int col = 0; col < 3; col++) {
        Player mark = model.getMarkAt(row, col);

        if(mark == null) {
          continue;
        }

        switch(mark) {
          case X:
            drawLine(g2d, row, col, row+1, col+1);
            drawLine(g2d, row, col+1, row+1, col);
            break;
          case O:
            drawOval(g2d, row, col, 1, 1);
            break;
        }
      }
    }
  }

  public void subscribe(ViewActions observer) {
    this.addMouseListener(new TTTMouseListener(observer));
  }

  class TTTMouseListener extends MouseAdapter {

    private ViewActions observer;

    public TTTMouseListener(ViewActions observer) {
      this.observer = observer;
    }

    public void mouseClicked(MouseEvent evt) {
      Point2D physical = evt.getPoint();

      try {
        AffineTransform physicalToLogical = getXformForLogicalToPhysical();
        physicalToLogical.invert();

        AffineTransform logicalToModel = getXFormForModelToLogical();
        logicalToModel.invert();

        Point2D logical = physicalToLogical.transform(physical, null);
        Point2D model = logicalToModel.transform(logical, null);
        observer.placeMark((int) model.getY(), (int) model.getX());
      } catch (NoninvertibleTransformException ex) {
        throw new RuntimeException(ex);
      }
    }

  }
}
