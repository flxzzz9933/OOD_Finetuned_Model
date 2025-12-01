package cs3500.lec09;

import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;

/**
 * An implementation of {@link IntSet1} using the Collections Framework's
 * {@link HashSet}. Yes, this is silly, but it's the quickest route to the
 * point of the demonstration.
 */
public class IntSet1Impl implements IntSet1 {
  private final Set<Integer> set = new HashSet<>();

  @Override
  public final void add(int value) {
    set.add(value);
  }

  @Override
  public final void remove(int value) {
    set.remove(value);
  }

  @Override
  public final boolean member(int value) {
    return set.contains(value);
  }

  @Override
  public final Iterator<Integer> iterator() {
    return set.iterator();
  }
}
