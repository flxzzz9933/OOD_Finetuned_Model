package cs3500.tictactoe;

import java.util.Arrays;
import java.util.stream.Collectors;

/**
 * In this representation, player X is 1 and player O is 2.
 */
public class TicTacToeModelVersionOne implements TicTacToe {

  protected final int[][] board;
  private int turn;

  public TicTacToeModelVersionOne() {
    board = new int[3][3];
    turn = 1;
  }

  @Override
  public void move(int row, int col) {
    if (isGameOver()) {
      throw new IllegalStateException("Game is over");
    }
    validateRowCol(row, col);
    if (board[row][col] != 0) {
      throw new IllegalArgumentException("Position occupied");
    }
    board[row][col] = turn;
    turn = (turn%2) + 1;
  }

  @Override
  public Player getTurn() {
    return getPlayerForInt(turn);
  }

  private Player getPlayerForInt(int player) {
    switch(player) {
      case 0:
        return null;
      case 1:
        return Player.X;
      case 2:
        return Player.O;
    }
    throw new RuntimeException("Unknown turn value");
  }

  @Override
  public boolean isGameOver() {
    boolean boardFull = true;
    for (int[] row : board) {
      if (Arrays.stream(row).anyMatch(cell -> cell == 0)) {
        boardFull = false;
        break;
      }
    }
    return boardFull || getWinner() != null;
  }

  @Override
  public Player getWinner() {
    for (int player : new int[] {1, 2}) {
      // check horizontals
      for (int[] row : board) {
        if(Arrays.stream(row).allMatch(mark -> mark == player)) {
          return getPlayerForInt(player);
        }
      }
      // check verticals
      for (int col = 0; col < board[0].length; col++) {
        if (board[0][col] == player && board[1][col] == player && board[2][col] == player) {
          return getPlayerForInt(player);
        }
      }
      // check diagonals
      if (board[0][0] == player && board[1][1] == player && board[2][2] == player) {
        return getPlayerForInt(player);
      }
      if (board[0][2] == player && board[1][1] == player && board[2][0] == player) {
        return getPlayerForInt(player);
      }
    }
    return null;
  }

  @Override
  public Player[][] getBoard() {
    Player[][] ret = new Player[3][3];
    for (int row = 0; row < board.length; row++) {
      for(int col = 0; col < board[0].length; col++) {
        ret[row][col] = getPlayerForInt(board[row][col]);
      }
    }
    return ret;
  }

  @Override
  public Player getMarkAt(int row, int col) {
    validateRowCol(row, col);
    return getPlayerForInt(board[row][col]);
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
