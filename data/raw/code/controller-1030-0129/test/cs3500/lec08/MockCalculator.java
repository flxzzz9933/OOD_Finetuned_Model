package cs3500.lec08;

public class MockCalculator implements ICalculator {

  private StringBuilder log;

  public MockCalculator(StringBuilder log) {
    this.log = log;
  }

  @Override
  public int add(int num1, int num2) {
    log.append(num1 + " " + num2);
    return 0;
  }
}
