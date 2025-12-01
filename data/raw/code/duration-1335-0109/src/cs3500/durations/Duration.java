package cs3500.durations;

public interface Duration {

  //Observations: methods that allow you to look into the state of an object

  /**
   * Convert this duration into a number of seconds equal to the duration
   * @return the number of seconds representing this duration
   */
  long inSeconds();

  /**
   * Represents this duration as HH:MM:SS.
   * For example, 181 seconds is 00:03:01.
   */
  String asHms();
}
