package cs3500.lec08;

import org.junit.Test;

import java.io.IOException;
import java.io.Reader;
import java.io.StringReader;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

public class TestController5 {

  //integration test: a test to check behavior of components working together
  @Test
  public void testRunCalculator() throws Exception {
    StringBuffer out = new StringBuffer();
    Reader in = new StringReader("3 4");
    CalcController controller5 = new Controller5(in, out);
    controller5.runCalculator(new Calculator());
    assertEquals("7\n", out.toString());
  }

  //unit test for the controller! helps check one of the jobs of the controller,
  //to pass input from the user correctly to the model as specified by the program
  @Test
  public void testControllerCorrectlyCallsAdd() throws IOException {
    Readable input = new StringReader("3\n4\n");
    StringBuilder output = new StringBuilder();
    StringBuilder log = new StringBuilder();
    CalcController controller = new Controller5(input, output);
    controller.runCalculator(new MockCalculator(log));
    //Want to assert we get "3 4" from the fake model
    assertEquals("3 4", log.toString());
  }

  @Test
  public void testFailingAppendable() {
    Readable input = new StringReader("3 4");
    Appendable output = new FailingAppendable();
    CalcController controller = new Controller5(input, output);
    assertThrows(IOException.class,
        () -> controller.runCalculator(new Calculator()));
  }
}
