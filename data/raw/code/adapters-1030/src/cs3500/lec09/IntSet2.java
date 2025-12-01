package cs3500.lec09;

import java.util.List;

public interface IntSet2 {
  /**
   * Inserts the elements of other into this set.
   */
  void unionWith(IntSet2 other);

  /**
   * Removes the elements of other from this set.
   */
  void differenceFrom(IntSet2 other);

  /**
   * Checks whether this set is a superset of another set.
   * @param other the other set
   * @return whether {@code other} is a subset of {@code this}
   */
  boolean isSupersetOf(IntSet2 other);

  /**
   * The contents of the set as a list of integers. Modifying the returned
   * list will have no effect on this set.
   * @return the list of integers in this set
   */
  List<Integer> asList();
}
