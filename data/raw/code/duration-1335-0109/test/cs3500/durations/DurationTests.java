package cs3500.durations;

import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class DurationTests {

  private Duration oneeighty;
  private Duration threemins;

  @Before
  public void setup() {
    this.oneeighty = new SimpleDuration(0, 0, 180);
    this.threemins = new SimpleDuration(0 ,3, 0);
  }

  @Test
  public void testInSeconds() {
    assertEquals(3 * 60, threemins.inSeconds());
  }

}
