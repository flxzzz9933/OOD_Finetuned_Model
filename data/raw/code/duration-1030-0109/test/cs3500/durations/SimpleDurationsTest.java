package cs3500.durations;

import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

public class SimpleDurationsTest {

  private Duration day;
  private Duration ninetySeconds;
  private Duration threeMinutes;

  //NOTE: This code does not compile yet because we haven't written this class.
  //      In your code, the class should at least have an empty constructor and
  //      implement the interface (i.e. class HMSDuration implements Duration)
  @Before
  public void setup() {
    day = new HMSDuration(24, 0 ,0);
  }

  @Test
  public void testInSeconds() {
    assertEquals(24 * 60 * 60, day.inSeconds());
  }
}
