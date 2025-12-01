package cs3500.tictactoe;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Implementation of a basic 3x3 game of TicTacToe. Game can end in a stalemate as well,
 * which is indicated by getWinner() returning null.
 */
public class TicTacToeModelVersionTwo implements TicTacToe {

  // The top left corner is 0,0. Grid is indexed by rows first, then columns.
  private List<List<Player>> grid;
  private Player turn;

  public TicTacToeModelVersionTwo() {
    this.grid = new ArrayList<>();
    for(int row = 0; row < 3; row++) {
      List<Player> rowList = new ArrayList<>();
      for(int col = 0; col < 3; col++) {
        rowList.add(null);
      }
      this.grid.add(rowList);
    }
    this.turn = Player.X;
  }

  @Override
  public String toString() {
    // Using Java stream API to save code:
    return Arrays.stream(getBoard()).map(
            row -> " " + Arrays.stream(row).map(
                p -> p == null ? " " : p.toString()).collect(Collectors.joining(" | ")))
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

  @Override
  public void move(int row, int col) {
    if(isGameOver()) {
      throw new IllegalStateException("Game is over");
    }
    if(row < 0 || row > 2 || col < 0 || col > 2) {
      throw new IllegalArgumentException("Bad row/col");
    }
    if(grid.get(row).get(col) != null) {
      throw new IllegalArgumentException("Already placed mark here");
    }
    grid.get(row).set(col, turn);
    if(turn == Player.X) {
      turn = Player.O;
    } else {
      turn = Player.X;
    }
  }

  @Override
  public Player getTurn() {
    return turn;
  }

  @Override
  public boolean isGameOver() {
    //every spot on board is filled
    //or a winner
    return isBoardFull() || isWinner();
  }

  private boolean isWinner() {
    return isThreeInARow() != null || isThreeInACol() != null || isThreeInADiagonal() != null;
  }

  //Slight change from class plan here: notice we will eventually need the winner
  //so we instead have methods to GET the possible winner. Then for isGameOver, we only
  //need to check if those winners are not null for the game to be over.
  private Player isThreeInARow() {
    for(int row = 0; row < 3; row++) {
      Player mark = grid.get(row).get(0);
      if (mark != null
          && grid.get(row).get(0) == mark
          && grid.get(row).get(1) == mark
          && grid.get(row).get(2) == mark) {
        return mark;
      }
    }
    return null;
  }

  private Player isThreeInACol() {
    for(int col = 0; col < 3; col++) {
      Player mark = grid.get(0).get(col);
      if (mark != null
          && grid.get(0).get(col) == mark
          && grid.get(1).get(col) == mark
          && grid.get(2).get(col) == mark) {
        return mark;
      }
    }
    return null;
  }

  private Player isThreeInADiagonal() {
    Player mark = grid.get(0).get(0);
    if (mark != null && grid.get(1).get(1) == mark && grid.get(2).get(2) == mark) {
      return mark;
    }
    mark = grid.get(0).get(2);
    if (mark != null && grid.get(1).get(1) == mark && grid.get(2).get(0) == mark) {
      return mark;
    }
    return null;
  }

  private boolean isBoardFull() {
    for(int row = 0; row < 3; row++) {
      for(int col = 0; col < 3; col++) {
        if (grid.get(row).get(col) == null) {
          return false;
        }
      }
    }
    return true;
  }

  @Override
  public Player getWinner() {
    Player winner = isThreeInARow();
    if(winner != null) return winner;

    winner = isThreeInACol();
    if(winner != null) return winner;

    return isThreeInADiagonal();
  }

  @Override
  public Player[][] getBoard() {
    //create the grid
    Player[][] copy = new Player[3][3];
    //put the contents of the grid into the new copy
    for(int row = 0; row < copy.length; row++) {
      for(int col = 0; col < copy[0].length; col++) {
        copy[row][col] = grid.get(row).get(col);
      }
    }
    //return the new copy
    return copy;
  }

  @Override
  public Player getMarkAt(int row, int col) {
    if (row < 0 || row > 2 || col < 0 || col > 2) {
      throw new IllegalArgumentException("Bad row/col");
    }
    return grid.get(row).get(col);
  }
}
