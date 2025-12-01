package cs3500.lec08;

import java.util.Scanner;

/**
 * Demonstrates a simple command-line-based calculator
 */
public class SimpleCalc2 {
  public static void main(String[] args) {
    int num1, num2;
    Scanner scan = new Scanner(System.in);
    num1 = scan.nextInt();
    num2 = scan.nextInt();
    System.out.printf("%d", new Calculator().add(num1, num2));
  }
}

interface ICalculator {
  int add(int num1, int num2);
}

class Calculator implements ICalculator {
  public int add(int num1, int num2) {
    return num1 + num2;
  }
}

