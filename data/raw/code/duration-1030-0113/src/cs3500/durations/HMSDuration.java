package cs3500.durations;

//NOTE: Implemented some of the other methods to focus on the important details next lecture
//      Mainly equals and plus.

public class HMSDuration implements Duration {

  private int hours;
  private int minutes;
  private int seconds;

  /**
   * Creates a non-negative duration with the given hours, minutes, seconds
   * @param hours number of hours in the duration
   * @param minutes number of minutes in the duration
   * @param seconds number of seconds in the duration
   * @throws IllegalArgumentException if hours, minutes, or seconds < 0
   */
  public HMSDuration(int hours, int minutes, int seconds) throws IllegalArgumentException {
    if (hours < 0 || minutes < 0 || seconds < 0) {
      throw new IllegalArgumentException("Bad arguments");
    }

    this.hours = hours;
    this.minutes = minutes;
    this.seconds = seconds;
    if(seconds >= 60) {
      this.minutes += seconds / 60;
      this.seconds = seconds % 60; //Get the remainder of a division
    }

    if(minutes >= 60) {
      this.hours += this.minutes / 60;
      this.minutes = this.minutes % 60;
    }

  }

  @Override
  public int inSeconds() {
    return this.hours * 3600 + this.minutes * 60 + this.seconds;
  }

  @Override
  public String asHms() {
    return String.format("%02d:%02d:%02d", hours, minutes, seconds);
  }

  @Override
  public int compareTo(Duration other) {
    return Long.compare(this.inSeconds(), other.inSeconds());
  }

  @Override
  public Duration plus(Duration other) {
    return null;
  }


}
