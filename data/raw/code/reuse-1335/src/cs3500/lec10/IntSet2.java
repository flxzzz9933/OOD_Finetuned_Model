package cs3500.lec10;

import java.util.HashSet;
import java.util.Set;

/**
 * An implementation of {@link IntSet} using the Collections Framework's
 * {@link HashSet}. Yes, this is silly, but it's the quickest route to the
 * point of the demonstration.
 *
 * @see InstrumentedIntSet2
 */
public class IntSet2 implements IntSet {
  private final Set<Integer> set = new HashSet<>();

  @Override
  public void add(int value) {
    set.add(value);
  }

  // In version 2, addAll() is not implemented in terms of add(). Subclasses
  // were overriding add() and causing trouble, so we decided to give in and
  // duplicate the code between add() and addAll(). This is okay, because
  // as the implementor of this class we are free to change the implementation
  // so long as we don't change the interface... right?
  @Override
  public void addAll(int... values) {
    for (int i : values) {
      set.add(i);
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
