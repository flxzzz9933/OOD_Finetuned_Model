package cs3500.lec09;

import java.util.Iterator;
import java.util.List;
import java.util.Objects;

/**
 * Adapts an {@link IntSet2} to the {@link IntSet1} interface. The is the
 * object adapter pattern.
 */
public final class IntSet2ToIntSet1Adapter implements IntSet1, IntSet2 {
  private final IntSet2 adaptee;

  public IntSet2ToIntSet1Adapter() {
    this(IntSet2Impl.empty());
  }

  public IntSet2ToIntSet1Adapter(IntSet2 adaptee) {
    Objects.requireNonNull(adaptee);
    this.adaptee = adaptee;
  }

  @Override
  public void add(int value) {
    unionWith(IntSet2Impl.singleton(value));
  }

  @Override
  public void remove(int value) {
    differenceFrom(IntSet2Impl.singleton(value));
  }

  @Override
  public boolean member(int value) {
    return isSupersetOf(IntSet2Impl.singleton(value));
  }

  @Override
  public Iterator<Integer> iterator() {
    return asList().iterator();
  }

  @Override
  public void unionWith(IntSet2 other) {
    adaptee.unionWith(other);
  }

  @Override
  public void differenceFrom(IntSet2 other) {
    adaptee.differenceFrom(other);
  }

  @Override
  public boolean isSupersetOf(IntSet2 other) {
    return adaptee.isSupersetOf(other);
  }

  @Override
  public List<Integer> asList() {
    return adaptee.asList();
  }
}
