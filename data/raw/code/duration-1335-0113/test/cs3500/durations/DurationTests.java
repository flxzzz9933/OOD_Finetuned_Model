package cs3500.durations;

import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

public class DurationTests {

  private Duration oneeighty;
  private Duration threemins;

  @Before
  public void setup() {
    oneeighty = new HMSDuration(0, 0, 180);
    threemins = new HMSDuration(0, 3, 0);
  }

  @Test
  public void testValidConstruction() {
    assertEquals("00:03:00", oneeighty.asHms());
    assertEquals(180, oneeighty.inSeconds());
  }

  @Test
  public void testInvalidConstructor() {
    assertThrows(IllegalArgumentException.class,
        () -> { new HMSDuration(-1, 0, 0); });
    assertThrows(IllegalArgumentException.class,
        () -> { new HMSDuration(0, -1, 0); });
  }

  @Test
  public void testInSeconds() {
    assertEquals(3 * 60, threemins.inSeconds());
  }

}
