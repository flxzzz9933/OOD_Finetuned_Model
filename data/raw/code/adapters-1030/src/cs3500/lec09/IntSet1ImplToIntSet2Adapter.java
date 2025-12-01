package cs3500.lec09;

import java.util.ArrayList;
import java.util.List;

/**
 * Adapts {@link IntSet1Impl} to the {@link IntSet2} interface. This is the
 * class adapter pattern.
 */
public class IntSet1ImplToIntSet2Adapter
  extends IntSet1Impl
  implements IntSet2
{
  /**
   * Constructs a new empty {@code IntSet1Adapter}.
   */
  public static IntSet1ImplToIntSet2Adapter empty() {
    return new IntSet1ImplToIntSet2Adapter();
  }

  /**
   * Constructs a new single-element {@code IntSet1Adapter}.
   */
  public static IntSet1ImplToIntSet2Adapter singleton(int i) {
    IntSet1ImplToIntSet2Adapter result = new IntSet1ImplToIntSet2Adapter();
    result.add(i);
    return result;
  }

  @Override
  public final void unionWith(IntSet2 other) {
    for (int i : other.asList()) {
      add(i);
    }
  }

  @Override
  public final void differenceFrom(IntSet2 other) {
    for (int i : other.asList()) {
      remove(i);
    }
  }

  @Override
  public final boolean isSupersetOf(IntSet2 other) {
    for (int z : other.asList()) {
      if (! member(z)) {
        return false;
      }
    }

    return true;
  }

  @Override
  public final List<Integer> asList() {
    List<Integer> result = new ArrayList<>();
    for (int i : this) {
      result.add(i);
    }
    return result;
  }
}
