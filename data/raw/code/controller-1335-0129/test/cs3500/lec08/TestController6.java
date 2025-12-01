package cs3500.lec08;

import org.junit.Test;

import java.io.Reader;
import java.io.StringReader;

import static org.junit.Assert.assertEquals;

public class TestController6 {
  @Test
  public void testRunCalculator() throws Exception {
    StringBuffer out = new StringBuffer();
    Reader in = new StringReader("+ 3 4 + 8 9 q");
    CalcController controller6 = new Controller6(in, out);
    controller6.runCalculator(new Calculator());
    assertEquals("7\n17\n", out.toString());
  }
}
