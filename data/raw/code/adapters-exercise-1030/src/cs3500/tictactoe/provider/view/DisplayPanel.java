package cs3500.tictactoe.provider.view;

import java.awt.*;
import java.awt.geom.AffineTransform;
import java.awt.geom.Point2D;

import javax.swing.*;

import cs3500.tictactoe.model.Player;

public class DisplayPanel extends JPanel {

  private Player[][] board;

  public DisplayPanel() {

  }

  @Override
  protected void paintComponent(Graphics g) {
    super.paintComponent(g);
    Graphics2D g2d = (Graphics2D)g.create();

    g2d.transform(transformFromLogicalToPhysical());

    //Two options:
    //Either set a stroke width so the lines for shapes
    //look nice
    //OR find a scalable constant to multiply every coordinate number by
    //to fit in the intermediate coordinates.
    //g2d.setStroke(new BasicStroke(0.05f));

    drawBoardState(g2d);
    drawGridLines(g2d);
  }

  public void updateBoard(Player[][] board) {
    this.board = board;
  }

  private void drawGridLines(Graphics2D g2d) {
    drawLine(g2d, new Point(1, 0), new Point(1, 3));
    drawLine(g2d, new Point(2, 0), new Point(2, 3));
    drawLine(g2d, new Point(0, 1), new Point(3, 1));
    drawLine(g2d, new Point(0, 2), new Point(3, 2));
  }



  public Dimension getPreferredLogicalSize() {
    return new Dimension(60, 60);
  }

  public Dimension getPreferredModelSize() {
    return new Dimension(3, 3);
  }

  private AffineTransform transformFromLogicalToPhysical() {
    AffineTransform xform = new AffineTransform();
    Dimension preferred = this.getPreferredLogicalSize();
    xform.scale(this.getWidth() / preferred.getWidth(),
        this.getHeight() / preferred.getHeight());
    return xform;
  }

  private AffineTransform transformModelToLogical() {
    AffineTransform xform = new AffineTransform();
    Dimension preferred = this.getPreferredLogicalSize();
    Dimension preferredModel = this.getPreferredModelSize();
    xform.scale(preferred.getWidth() / preferredModel.getWidth(),
        preferred.getHeight() / preferredModel.getHeight());
    return xform;
  }

  //Draw the board state
  private void drawBoardState(Graphics2D g2d) {
    if(board != null) {
      for (int row = 0; row < board.length; row++) {
        for (int col = 0; col < board[0].length; col++) {
          if (board[row][col] != null) {
            drawPlayer(g2d, row, col, board[row][col]);
          }
        }
      }
    }
  }

  private void drawPlayer(Graphics2D g2d, int row, int col, Player player) {
    Color oldColor = g2d.getColor();

    switch(player) {
      case X:
        g2d.setColor(Color.CYAN);
        drawLine(g2d, new Point(col, row), new Point(col+1, row+1));
        drawLine(g2d, new Point(col+1, row), new Point(col, row+1));
        break;
      case O:
        g2d.setColor(Color.MAGENTA);
        drawOval(g2d, new Point(col, row), 0.9, 0.9);
        break;
      default:
        //draw nothing
        break;
    }

    g2d.setColor(oldColor);

  }

  private void drawLine(Graphics2D g2d, Point2D src, Point2D dst) {
    Point2D logicalSrc = transformModelToLogical().transform(src, null);
    Point2D logicalDst = transformModelToLogical().transform(dst, null);
    g2d.drawLine((int)logicalSrc.getX(),
        (int)logicalSrc.getY(),
        (int)logicalDst.getX(),
        (int)logicalDst.getY());
  }

  private void drawOval(Graphics2D g2d, Point leftCorner, double width, double height) {
    Point2D logicalLeftCorner = transformModelToLogical().transform(leftCorner, null);
    Point2D logicalDimensions = transformModelToLogical().transform(new Point2D.Double(width, height), null);
    g2d.drawOval((int)logicalLeftCorner.getX(), (int)logicalLeftCorner.getY(),
        (int)logicalDimensions.getX(), (int)logicalDimensions.getY());
  }
}
