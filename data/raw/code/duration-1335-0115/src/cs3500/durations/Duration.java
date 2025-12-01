package cs3500.durations;

/**
 * Behaviors necessary for a Duration.
 * We assume durations are non-negative and can be read as seconds.
 */
public interface Duration extends Comparable<Duration> {

  //OBSERVATIONS: methods that allow you to look into the state of an object

  /**
   * Convert this duration into a number of seconds equal to the duration
   * @return the number of seconds representing this duration
   */
  long inSeconds();

  /**
   * Represents this duration as H:MM:SS. Minutes and seconds are zero-padded,
   * hours can be as large as needed, but must be at least 2 digits and zero-padded.
   * For example, 181 seconds is 00:03:01.
   * @return String representation of the duration
   */
  String asHms();

  /**
   * Creates a new Duration that is the sum of this duration of time and
   * another duration of time.
   * @param other the other Duration to add
   * @return sum of two Durations, this one and other
   */
  Duration plus(Duration other);






}
