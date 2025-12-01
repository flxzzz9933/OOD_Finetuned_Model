package cs3500.durations;

/**
 * A Duration that stores time in hours, minutes, and seconds.
 * Ensures seconds and minutes are between 0 and 59 inclusive.
 */
public class HMSDuration implements Duration {

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

  @Override
  public long inSeconds() {
    return 0;
  }

  @Override
  public String asHms() {
    return String.format("%02d:%02d:%02d", hours, minutes, seconds);
  }

  @Override
  public Duration plus(Duration other) {
    return null;
  }

  @Override
  public int compareTo(Duration o) {
    return 0;
  }
}
