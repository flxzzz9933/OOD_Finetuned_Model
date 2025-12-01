package cs3500.lec10;

import java.util.HashSet;
import java.util.Set;

/**
 * One way to fix the problems with {@link InstrumentedIntSet1} and {@link
 * InstrumentedIntSet2} is to make sure that the superclass ({@code IntSet3})
 * doesn't call any methods that the subclass ({@link InstrumentedIntSet3})
 * can override. Thus, instead of having {@link #addAll(int...)} call {@link
 * #add}, we factor out the common functionality into a private method {@link
 * #_add(int)} and have both public methods call that.
 *
 * <p>Since our class allows extension (i.e., isn't final), we need to
 * <i>document</i> what it means to subclass it. In this case we might write
 * something like this:
 *
 * <p><b>Inheritance note:</b> It is safe for subclasses to override any and
 * all methods of this class, since none of the methods calls a public method
 * that might be overridden.
 *
 * @see InstrumentedIntSet3
 */
public class IntSet3 implements IntSet {
  private final Set<Integer> set = new HashSet<>();

  // Here's the private method to factor out add() and addAll()'s common code:
  private void _add(int value) {
    set.add(value);
  }

  // Now add() is implemented using _add():
  @Override
  public void add(int value) {
    _add(value);
  }

  // And so is addAll():
  @Override
  public void addAll(int... values) {
    for (int i : values) {
      _add(i);
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
