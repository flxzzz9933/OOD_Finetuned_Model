package cs3500.turtle.view;

import java.awt.event.ActionListener;
import java.awt.event.KeyListener;
import java.util.List;
import java.util.function.Consumer;

import cs3500.turtle.model.Position2D;
import cs3500.turtle.tracingmodel.Line;

/**
 * The view interface. To motivate the methods here
 * think about all the operations that the controller
 * would need to invoke on the view
 */
public interface IView {
  /**
   * Make the view visible. This is usually called
   * after the view is constructed
   */
  void makeVisible();

  /**
   * Transmit an error message to the view, in case
   * the command could not be processed correctly
   *
   * @param error
   */
  void showErrorMessage(String error);

  /**
   * Signal the view to draw itself
   */
  void refresh();

  /*
   TODO: Add a subscribe method for the actions of the view
   */
  void setViewActions(ViewActions actions);
    
  /*
   TODO: Add methods here to tell the view where the current turtle
   is, which way it's facing, and what lines should be drawn.
  */
}
