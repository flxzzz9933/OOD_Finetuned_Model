package cs3500.tictactoe;

import java.util.Arrays;
import java.util.Objects;
import java.util.stream.Collectors;

/**
 * Represents a single game of Tic Tac Toe. Three in a row down/across/diagonally to win; X goes
 * first.
 */
public class TicTacToeModel implements TicTacToe {
  private final Player[][] board;
  private Player turn;

  public TicTacToeModel() {
    board = new Player[3][3];
    turn = Player.X;
  }

  @Override
  public void move(int row, int col) {
    if (isGameOver()) {
      throw new IllegalStateException("Game is over");
    }
    validateRowCol(row, col);
    if (board[row][col] != null) {
      throw new IllegalArgumentException("Position occupied");
    }
    board[row][col] = turn;
    turn = turn == Player.X ? Player.O : Player.X;
  }

  @Override
  public Player getTurn() {
    return turn;
  }

  @Override
  public boolean isGameOver() {
    boolean boardFull = true;
    for (Player[] row : board) {
      if (Arrays.stream(row).anyMatch(Objects::isNull)) {
        boardFull = false;
        break;
      }
    }
    return boardFull || getWinner() != null;
  }

  @Override
  public Player getWinner() {
    for (Player player : Player.values()) {
      // check horizontals
      for (Player[] row : board) {
        if(Arrays.stream(row).allMatch(mark -> mark == player)) {
          return player;
        }
      }
      // check verticals
      for (int col = 0; col < board[0].length; col++) {
        if (board[0][col] == player && board[1][col] == player && board[2][col] == player) {
          return player;
        }
      }
      // check diagonals
      if (board[0][0] == player && board[1][1] == player && board[2][2] == player) {
        return player;
      }
      if (board[0][2] == player && board[1][1] == player && board[2][0] == player) {
        return player;
      }
    }
    return null;
  }

  @Override
  public Player[][] getBoard() {
    Player[][] ret = new Player[3][3];
    for (int row = 0; row < board.length; row++) {
      ret[row] = Arrays.copyOf(board[row], board[row].length);
    }
    return ret;
  }

  @Override
  public Player getMarkAt(int row, int col) {
    validateRowCol(row, col);
    return board[row][col];
  }

  //NOTE: Could be made non-static as well.
  //Made static here to show that there is no dynamic information needed. However, it can't be used
  //outside of this class due to it being private.
  private static void validateRowCol(int row, int col) {
    if (row < 0 || row > 2 || col < 0 || col > 2) {
      throw new IllegalArgumentException("Invalid board position: " + row + "," + col);
    }
  }

  @Override
  public String toString() {
    // Using Java stream API to save code:
    return Arrays.stream(getBoard())
        .map(row -> " " + Arrays.stream(row)
            .map(player -> player == null ? " " : player.toString())
            .collect(Collectors.joining(" | ")))
          .collect(Collectors.joining("\n-----------\n"));
    // This is the equivalent code as above, but using iteration, and still using the helpful
    // built-in String.join method.
    // List<String> rows = new ArrayList<>();
    // for(Player[] row : getBoard()) {
    //   List<String> rowStrings = new ArrayList<>();
    //   for(Player p : row) {
    //     if(p == null) {
    //       rowStrings.add(" ");
    //     } else {
    //       rowStrings.add(p.toString());
    //     }
    //   }
    //   rows.add(" " + String.join(" | ", rowStrings));
    // }
    // return String.join("\n-----------\n", rows);
  }
}
