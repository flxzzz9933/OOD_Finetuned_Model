package cs3500.lec09;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * An implementation of the alternative {@link IntSet2} interface, using
 * the Collections Framework's {@link HashSet} as {@link IntSet1Impl}
 * does.
 */
public final class IntSet2Impl implements IntSet2 {
  private final Set<Integer> set = new HashSet<>();

  private IntSet2Impl() { }

  /**
   * Constructs a new empty {@code IntSet2}.
   */
  public static IntSet2 empty() {
    return new IntSet2Impl();
  }

  /**
   * Constructs a new single-element {@code IntSet2}.
   */
  public static IntSet2 singleton(int i) {
    IntSet2Impl result = new IntSet2Impl();
    result.set.add(i);
    return result;
  }

  @Override
  public void unionWith(IntSet2 other) {
    set.addAll(other.asList());
  }

  @Override
  public void differenceFrom(IntSet2 other) {
    set.removeAll(other.asList());
  }

  @Override
  public boolean isSupersetOf(IntSet2 other) {
    for (int z : other.asList()) {
      if (! set.contains(z)) {
        return false;
      }
    }

    return true;
  }

  @Override
  public List<Integer> asList() {
    return new ArrayList<>(set);
  }
}
