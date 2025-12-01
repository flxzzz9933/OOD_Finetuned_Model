package cs3500.tictactoe.view;

/**
 * A view for Tic-Tac-Toe: display the game board and provide visual interface for users.
 */
public interface TTTView {

  /**
   * Set up the controller to handle click events in this view.
   * @param listener the controller
   */
  void addClickListener(ViewActions listener);

  /**
   * Refresh the view to reflect any changes in the game state.
   */
  void refresh();

  /**
   * Make the view visible to start the game session.
   */
  void makeVisible();
}
