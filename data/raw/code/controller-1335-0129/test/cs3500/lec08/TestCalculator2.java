package cs3500.lec08;

import org.junit.Test;

import java.io.IOException;
import java.io.StringReader;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

public class TestCalculator2 {
  @Test
  public void testAdd() {
    assertEquals(7, new Calculator().add(3, 4));
  }

  @Test
  public void testController() {
    //2. How do I give my test input so it runs automatically?
    //Answer: Make the Controller take in a Readable
    //  Then create a StringReader in the test.
    StringReader input = new StringReader("3\n4\n");

    //1. How do I even get the output so I can write an assert?
    //Answer: Make the controller take in an Appendalbe
    //  Then create a StringBuilder in the test and pass it into the controller.
    StringBuilder output = new StringBuilder();
    CalcController controller = new Controller3(input, output);
    try {
      controller.runCalculator(new Calculator());
      assertEquals("7", output.toString());
    } catch (IOException ex) {
      fail("Sent exception when not expected");
    }
  }
}
