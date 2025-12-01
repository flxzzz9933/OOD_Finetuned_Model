package cs3500.lec10;

/**
 * An integer set equipt with instrumentation to count the number of
 * adds. Method {@link #getAddCount()} lets the client find out the
 * result of the counting.
 */
public interface InstrumentedIntSet extends IntSet {
  /**
   * Returns the count of how many times an element has been added to the set.
   *
   * @return the count of added elements
   */
  int getAddCount();
}
