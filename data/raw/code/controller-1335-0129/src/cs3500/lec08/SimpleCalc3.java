package cs3500.lec08;

import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Objects;
import java.util.Scanner;

/**
 * Demonstrates a simple command-line-based calculator
 */
public class SimpleCalc3 {
  public static void main(String[] args) {
    try {
      new Controller3(new InputStreamReader(System.in), System.out)
          .runCalculator(new Calculator());
    } catch (IOException ex) {
      throw new RuntimeException("Something has gone wrong");
    }
  }
}

class Controller3 implements CalcController {
  private Readable input;
  private Appendable output;

  public Controller3(Readable input, Appendable output) {
    this.input = input;
    this.output = output;
  }

  public void runCalculator(ICalculator calc) throws IOException {
    Objects.requireNonNull(calc);
    int num1, num2;
    Scanner scan = new Scanner(input);
    num1 = scan.nextInt();
    num2 = scan.nextInt();
    String formattedOutput = String.format("%d", calc.add(num1, num2));
    output.append(formattedOutput);
  }
}

