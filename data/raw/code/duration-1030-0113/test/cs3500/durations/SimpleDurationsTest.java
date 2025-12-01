package cs3500.durations;

import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
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
    ninetySeconds = new HMSDuration(0, 0, 90);
    threeMinutes = new HMSDuration(0, 3, 00);
  }

  @Test
  public void testValidConstruction() {
    assertEquals("24:00:00", day.asHms());
    assertEquals("00:01:30", ninetySeconds.asHms());
    assertEquals("100:00:00", new HMSDuration(100, 0, 0).asHms());
  }

  @Test
  public void testInvalidConstruction() {
    assertThrows(IllegalArgumentException.class,
        () -> { new HMSDuration(-1, 0,0 ); }); //line 35 is a LAMBDA
  }

  @Test
  public void testInSeconds() {
    assertEquals(24 * 60 * 60, day.inSeconds());
  }
}
