package cs3500.lec10;

import java.util.HashSet;
import java.util.Set;

/**
 * Another way to fix the problems with {@link InstrumentedIntSet1} and {@link
 * InstrumentedIntSet2} is to make sure that any methods called by other
 * methods of the class are final, which prevents subclasses from messing
 * with them. The easy way to do this is to make this whole class final,
 * which makes inheritance not a concern. Unless we design our class for
 * extension and document how to extend it, making the class final is the
 * right thing to do. It may inconvenience someone who wants to extend our
 * class, but they can always use delegation.
 *
 * @see InstrumentedIntSet4
 */
public final class IntSet4 implements IntSet {
  private final Set<Integer> set = new HashSet<>();

  @Override
  public void add(int value) {
    set.add(value);
  }

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
