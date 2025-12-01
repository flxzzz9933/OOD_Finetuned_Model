package cs3500.lec10;

import java.util.HashSet;
import java.util.Set;

/**
 * An implementation of {@link IntSet} using the Collections Framework's
 * {@link HashSet}. Yes, this is silly, but it's the quickest route to the
 * point of the demonstration.
 *
 * @see InstrumentedIntSet1
 */
public class IntSet1 implements IntSet {
  private final Set<Integer> set = new HashSet<>();

  @Override
  public void add(int value) {
    set.add(value);
  }

  // addAll() is implemented in terms of add() in order avoid code duplication.
  // In this example the implementation of add() is minuscule, but in a real
  // implementation it might be complicated, and we wouldn't want to repeat
  // it in addAll() as well.
  @Override
  public void addAll(int... values) {
    for (int i : values) {
      add(i);
    }
  }

  @Override
  public void remove(int value) {
    set.remove(value);
  }

  @Override
  public boolean member(int value) {
    return set.contains(value);
  }
}
