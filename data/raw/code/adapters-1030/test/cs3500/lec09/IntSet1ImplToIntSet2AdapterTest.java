package cs3500.lec09;

import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class IntSet1ImplToIntSet2AdapterTest {
  IntSet1ImplToIntSet2Adapter set1 = new IntSet1ImplToIntSet2Adapter();
  IntSet1ImplToIntSet2Adapter set2 = IntSet1ImplToIntSet2Adapter.singleton(5);

  @Before
  public void setUp() throws Exception {
    set1.add(2);
    set1.unionWith(sing(3));
  }

  @Test
  public void memberSeesAdd() {
    assertTrue(set1.member(2));
  }

  @Test
  public void memberSeesUnionWith() {
    assertTrue(set1.member(3));
  }

  @Test
  public void memberSeesSingleton() {
    assertTrue(set2.member(5));
  }

  @Test
  public void memberCanReturnFalse() {
    assertFalse(set2.member(2));
    assertFalse(set2.member(3));
    assertFalse(set2.member(4));
    assertFalse(set1.member(5));
  }

  @Test
  public void isSupersetOfSeesAdd() {
    assertTrue(set1.isSupersetOf(sing(2)));
  }

  @Test
  public void isSupersetOfSeesUnionWith() {
    assertTrue(set1.isSupersetOf(sing(3)));
  }

  @Test
  public void isSupersetOfSeesSingleton() {
    assertTrue(set2.isSupersetOf(sing(5)));
  }

  @Test
  public void isSupersetOfCanReturnFalse() {
    assertFalse(set2.isSupersetOf(sing(2)));
    assertFalse(set2.isSupersetOf(sing(3)));
    assertFalse(set2.isSupersetOf(sing(4)));
    assertFalse(set1.isSupersetOf(sing(5)));
  }

  @Test
  public void removeRemoves() {
    set2.remove(2);
    assertFalse(set2.member(2));
    assertTrue(set2.member(2));
  }

  // No tests for iterator or asList yet.

  public static IntSet2 sing(int value) {
    return IntSet2Impl.singleton(value);
  }
}
