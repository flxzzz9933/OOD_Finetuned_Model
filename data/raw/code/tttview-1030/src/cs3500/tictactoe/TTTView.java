package cs3500.tictactoe;

/**
 * A view for Tic-Tac-Toe: display the game board and provide visual interface for users.
 */
public interface TTTView {

  /**
   * Refresh the view to reflect any changes in the game state.
   */
  void refresh();

  /**
   * Make the view visible to start the game session.
   */
  void makeVisible();

  /**
   * Adds the observer to any listeners so actions on the view are
   * delegated to the observer.
   * @param observer
   */
  void subscribe(ViewActions observer);
}
