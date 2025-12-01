package cs3500.durations;

/**
 * Behaviors for a Duration, a length of time.
 * We assume all Durations have a minimum resolution of seconds, meaning
 * we can always convert to seconds as needed.
 */
public interface Duration {
  //OBSERVATIONS: Behaviors that look into the state of an object

  /**
   * Converts the given duration from whatever units it knows into seconds
   * @return the number of seconds equal to the given duration
   */
  int inSeconds();

  /**
   * Returns the given duration of time as a string in the format of
   * HH:MM:SS. For example, 180 seconds is formatted as
   * 00:01:30.
   * @return the above String representation of this Duration
   */
  String asHms();

}
