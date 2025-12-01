package cs3500.lec08;

import org.junit.Test;

import java.io.IOException;
import java.io.Reader;
import java.io.StringReader;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

public class TestController5 {

  //integration test: a test that checks the behavior of components working together
  @Test
  public void testRunCalculator() throws Exception {
    StringBuffer out = new StringBuffer();
    Reader in = new StringReader("3 4");
    CalcController controller5 = new Controller5(in, out);
    controller5.runCalculator(new Calculator());
    assertEquals("7\n", out.toString());
  }

  @Test
  public void testInputs() throws IOException {
    StringReader in = new StringReader("3 4");
    StringBuilder out = new StringBuilder();
    CalcController controller = new Controller5(in, out);
    StringBuilder log = new StringBuilder();
    controller.runCalculator(new MockCalculator(log));
    assertEquals("3 4", log.toString());
  }

  @Test
  public void testFailingAppendable() {
    StringReader in = new StringReader("3 4");
    Appendable out = new FailingAppendable();
    CalcController controller = new Controller5(in, out);
    assertThrows(IOException.class,
        () -> controller.runCalculator(new Calculator()));
  }
}
