package cs3500.durations;

import java.util.Objects;

public abstract class AbstractDuration implements Duration {

  @Override
  public Duration plus(Duration other) {
    return fromSeconds(this.inSeconds() + other.inSeconds());
  }

  //protected means any subclass of this class can see it
  protected abstract Duration fromSeconds(long seconds);

  @Override
  public int compareTo(Duration other) {
    return Long.compare(this.inSeconds(), other.inSeconds());
  }

  @Override
  public boolean equals(Object other) {
    if (!(other instanceof Duration)) {
      return false;
    }
    Duration that = (Duration) other;
    return this.compareTo(that) == 0;
    //return this.inSeconds() == that.inSeconds();
  }

  @Override
  public int hashCode() {
    return Objects.hash(this.inSeconds());
  }


}
