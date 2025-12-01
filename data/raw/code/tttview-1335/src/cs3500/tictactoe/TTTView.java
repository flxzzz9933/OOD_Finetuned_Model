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

  void subscribe(ViewActions observer);
}
