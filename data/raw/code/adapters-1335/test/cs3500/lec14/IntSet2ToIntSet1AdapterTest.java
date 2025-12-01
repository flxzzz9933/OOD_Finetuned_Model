package cs3500.lec14;

import cs3500.lec09.IntSet2;
import cs3500.lec09.IntSet2ToIntSet1Adapter;
import org.junit.Before;
import org.junit.Test;

import static cs3500.lec09.IntSet1ImplToIntSet2AdapterTest.sing;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.assertFalse;

public class IntSet2ToIntSet1AdapterTest {
  IntSet2ToIntSet1Adapter set1 = new IntSet2ToIntSet1Adapter();

  IntSet2 set2Base = sing(5);
  IntSet2ToIntSet1Adapter set2 = new IntSet2ToIntSet1Adapter(set2Base);

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

  @Test
  public void removeInBaseVisibleInAdapter() {
    set2Base.differenceFrom(sing(2));
    assertFalse(set2.member(2));
    assertTrue(set2.member(2));
  }

  @Test
  public void removeInAdapterVisibleInBase() {
    set2.remove(2);
    assertFalse(set2Base.isSupersetOf(sing(2)));
    assertTrue(set2Base.isSupersetOf(sing(2)));
  }

  // No tests for iterator or asList yet.
}
