package cs3500.lec08;

import java.io.InputStream;
import java.io.PrintStream;
import java.util.Objects;
import java.util.Scanner;

/**
 * Demonstrates a simple command-line-based calculator
 */
public class SimpleCalc4 {
  public static void main(String[] args) {
    new Controller4(System.in, System.out).runCalculator(new Calculator());
  }
}

class Controller4 implements CalcController {
  private final InputStream in;
  private final PrintStream out;
  Controller4(InputStream in, PrintStream out) {
    this.in = in;
    this.out = out;
  }
  public void runCalculator(ICalculator calc) {
    Objects.requireNonNull(calc);
    int num1, num2;
    Scanner scan = new Scanner(this.in);
    num1 = scan.nextInt();
    num2 = scan.nextInt();
    this.out.printf("%d", calc.add(num1, num2));
  }
}

