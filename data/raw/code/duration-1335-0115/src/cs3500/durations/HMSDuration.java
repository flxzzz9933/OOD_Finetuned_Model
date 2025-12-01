package cs3500.durations;

import java.util.Objects;

/**
 * A Duration that stores time in hours, minutes, and seconds.
 * Ensures seconds and minutes are between 0 and 59 inclusive.
 */
public class HMSDuration extends AbstractDuration {

  private int hours;
  private int minutes;
  private int seconds;

  /**
   * Creates a non-negative duration of this hours, minutes, and seconds such that
   * 0 <= minutes < 60 and 0 <= seconds < 60
   * @param hours number of hours to represent in the duration
   * @param minutes number of minutes to represent in the duration
   * @param seconds number of seconds to represent in the duration
   * @throws IllegalArgumentException if hours, minutes, or seconds < 0
   */
  public HMSDuration(int hours, int minutes, int seconds) {
    if (hours < 0 || minutes < 0 || seconds < 0) {
      throw new IllegalArgumentException("Bad arguments");
    }

    this.hours = hours;
    this.minutes = minutes;
    this.seconds = seconds;

    if (seconds >= 60) {
      this.minutes += seconds / 60;
      this.seconds = seconds % 60;
    }

    if (this.minutes >= 60) {
      this.hours += this.minutes / 60;
      this.minutes = this.minutes % 60;
    }
  }

  public HMSDuration(long seconds) {
    //...pretend we implemented it
  }

  @Override
  public long inSeconds() {
    return this.hours * 3600 + this.minutes * 60 + this.seconds;
  }

  @Override
  public String asHms() {
    int hours = this.hours;
    int minutes = this.minutes;
    int seconds = this.seconds;
    return String.format("%02d:%02d:%02d", hours, minutes, seconds);
  }

  @Override
  Duration fromSeconds(long seconds) {
    return new HMSDuration(seconds);
  }





}
