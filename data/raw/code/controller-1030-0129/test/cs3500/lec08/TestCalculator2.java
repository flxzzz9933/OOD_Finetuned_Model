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
    //I can't even enter input!!
    //Answer: Have the controller take in a Readable
    //  And in the test, create a StringReader with the input you want
    Readable input = new StringReader("3\n4\n");
    //For output: Have the controller take in an Appendable
    //  And in the test, create a StringBuilder to store that output...
    StringBuilder output = new StringBuilder();
    Controller3 controller = new Controller3(input, output);
    try {
      controller.runCalculator(new Calculator());
      //... We can use toString on the builder to get that output.
      // And we can do whatever asserts we want on it!
      assertEquals("7", output.toString());
      //What do i assert here? The program just ends...
    } catch (IOException ex) {
      fail("I/O error occured when it shouldn't");
    }
  }
}
