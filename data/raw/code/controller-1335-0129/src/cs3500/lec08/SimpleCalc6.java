package cs3500.lec08;

import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Objects;
import java.util.Scanner;

/**
 * Demonstrates a simple command-line-based calculator
 */
public class SimpleCalc6 {
  public static void main(String[] args) {
    try {
      new Controller6(new InputStreamReader(System.in), System.out).runCalculator(new Calculator());
    } catch (IOException e) {
      e.printStackTrace();
    }
  }
}

class Controller6 implements CalcController {
  final Readable in;
  final Appendable out;
  Controller6(Readable in, Appendable out) {
    this.in = in;
    this.out = out;
  }
  public void runCalculator(ICalculator calc) throws IOException {
    Objects.requireNonNull(calc);
    int num1, num2;
    Scanner scan = new Scanner(this.in);
    while (true) {
      switch (scan.next()) {
        case "+":
          num1 = scan.nextInt();
          num2 = scan.nextInt();
          this.out.append(String.format("%d\n", calc.add(num1, num2)));
          break;
        case "q":
          return;
      }
    }
  }
}

