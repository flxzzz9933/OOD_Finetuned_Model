package cs3500.lec10;

import java.util.Objects;

/**
 * In this case, the author of {@link IntSet4} decided to avoid the
 * inheritance problem by making {@code IntSet4} final. That means we cannot
 * extend it, so instead we use the delegate pattern. The idea is that
 * instead of extending {@code IntSet4}, we keep one in a field. Then we can
 * implement our {@link InstrumentedIntSet} by delegating method calls to the
 * delegate (that is, letting the delegate do the work), and we can extend
 * the functionality by doing other things in our wrapper methods besides
 * delegating (such as updating the count).
 *
 * <p>One disadvantage of this approach is that we now have to write a shim
 * for every method, even if all it does is delegate, whereas inheritance
 * would have included all the non-overridden methods for us. An advantage is
 * the increase in flexibility, since we can now do things such as instrument
 * an existing {@link IntSet} without having to know its implementation.
 * Since the delegate is a field, we could also do things like change it at
 * will. (This class doesn't support that at this point, but it wouldn't be
 * difficult to add.)
 */
public class InstrumentedIntSet4 implements InstrumentedIntSet {
  private int addCount = 0;
  private final IntSet delegate;

  /**
   * Constructs a new instrumented integer set.
   */
  public InstrumentedIntSet4() {
    delegate = new IntSet4();
  }

  /**
   * Constructs an instrumented integer set on top of an existing
   * {@link IntSet}.
   *
   * @param base the integer set to instrument (non-null)
   */
  public InstrumentedIntSet4(IntSet base) {
    Objects.requireNonNull(base, "base cannot be null");
    delegate = base;
  }

  /**
   * Returns the count of how many times an element has been added to the set.
   *
   * @return the count of added elements
   */
  public int getAddCount() {
    return addCount;
  }

  @Override
  public void add(int value) {
    delegate.add(value);
    ++addCount;
  }

  @Override
  public void addAll(int... values) {
    delegate.addAll(values);
    addCount += values.length;
  }

  @Override
  public void remove(int value) {
    delegate.remove(value);
  }

  @Override
  public boolean member(int value) {
    return delegate.member(value);
  }
}
