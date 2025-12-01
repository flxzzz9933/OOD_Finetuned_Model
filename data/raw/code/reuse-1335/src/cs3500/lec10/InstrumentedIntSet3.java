package cs3500.lec10;

/**
 * An integer set that is instrumented to count the number of times an
 * element is added. This time it works because the superclass never calls
 * methods that the subclass might override, <b>and</b> as the implementor of
 * the subclass, we can be confident of this because the superclass {@link
 * IntSet3} documents this fact.
 */
public class InstrumentedIntSet3
  extends IntSet3
  implements InstrumentedIntSet
{
  private int addCount = 0;

  @Override
  public int getAddCount() {
    return addCount;
  }

  @Override
  public void add(int value) {
    super.add(value);
    ++addCount;
  }

  @Override
  public void addAll(int... values) {
    super.addAll(values);
    addCount += values.length;
  }
}
