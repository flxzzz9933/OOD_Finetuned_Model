package cs3500.tictactoe;

import java.io.IOException;
import java.util.InputMismatchException;
import java.util.Scanner;

public class TicTacToeConsoleController implements TicTacToeController {

  private final Appendable out;
  private final Scanner scan;

  public TicTacToeConsoleController(Readable in, Appendable out) {
    if (in == null || out == null) {
      throw new IllegalArgumentException("Readable and Appendable can't be null");
    }
    this.out = out;
    scan = new Scanner(in);
  }

  @Override
  public void playGame(TicTacToe model) {
    try {
      while(!model.isGameOver()) {
        printGameState(model);
        printPrompt(model);
        readInputs(model);
      }
    } catch (IOException ex) {
      throw new IllegalStateException("Bad IO");
    }
  }

  private void transmit(String message) throws IOException {
    out.append(message + "\n");
  }

  private void printGameState(TicTacToe model) throws IOException {
    transmit(model.toString());
  }

  private void printPrompt(TicTacToe model) throws IOException {
    transmit("Enter a move for " + model.getTurn() + ":");
  }

  private void readInputs(TicTacToe model) {
    int row;
    row = readInt();
    int col = readInt();
    model.move(row, col);
  }

  private int readInt() {
    int ans;
    try {
      ans = scan.nextInt() - 1;
    } catch (InputMismatchException ex) {
      String token = scan.next();
      if(token.equals("q") || token.equals("Q")) {
        //How should I signal to playGame to quit the method?
        //1. Change some field to true and have every method react to that
        //2. Return -1 to indicate the game is over (and propagate that message)
        //3. Throw an exception and handle it when ready
      }
      ans = readInt();
    }
    return ans;
  }

}
