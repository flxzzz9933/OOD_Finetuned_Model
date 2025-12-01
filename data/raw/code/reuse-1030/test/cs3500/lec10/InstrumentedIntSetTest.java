package cs3500.lec10;

import org.junit.Test;

import static org.junit.Assert.assertEquals;

/**
 * Tests for instrumented integer sets. All four implementations ought to
 * pass the same tests (if they are correct, which two of them aren't).
 * We use the factory method pattern to allow the abstract base class to
 * define the tests once, and then each four subclasses overrides the factory
 * method to return an instance of the class to test.
 */
public abstract class InstrumentedIntSetTest {
  protected abstract InstrumentedIntSet factory();

  @Test
  public final void testAdd() {
    InstrumentedIntSet set = factory();

    set.add(1);
    set.add(2);
    set.add(3);
    assertEquals(3, set.getAddCount());
  }

  @Test
  public final void testAddAll() {
    InstrumentedIntSet set = factory();

    set.addAll(1, 2, 3);
    assertEquals(3, set.getAddCount());
  }

  /**
   * One of these tests fails. Can you figure out why?
   */
  public static final class Test1 extends InstrumentedIntSetTest {
    protected InstrumentedIntSet factory() {
      return new InstrumentedIntSet1();
    }
  }

  /**
   * One of these tests <i>still</i> fails. Can you figure out why?
   */
  public static final class Test2 extends InstrumentedIntSetTest {
    protected InstrumentedIntSet factory() {
      return new InstrumentedIntSet2();
    }
  }

  /**
   * These should succeed.
   */
  public static final class Test3 extends InstrumentedIntSetTest {
    protected InstrumentedIntSet factory() {
      return new InstrumentedIntSet3();
    }
  }

  /**
   * These should succeed.
   */
  public static final class Test4 extends InstrumentedIntSetTest {
    protected InstrumentedIntSet factory() {
      return new InstrumentedIntSet4();
    }
  }

  /**
   * Here we see that with the delegate pattern approach, we can use any
   * implementation of {@link IntSet}, even one that caused us trouble before
   * ({@link IntSet1}) as the delegate. No inheritance, no problem.
   *
   * <p>Here the delegate is an {@link IntSet1}.
   */
  public static final class Test4Using1 extends InstrumentedIntSetTest {
    protected InstrumentedIntSet factory() {
      return new InstrumentedIntSet4(new IntSet1());
    }
  }

  /**
   * Here the delegate is an {@link IntSet2}.
   */
  public static final class Test4Using2 extends InstrumentedIntSetTest {
    protected InstrumentedIntSet factory() {
      return new InstrumentedIntSet4(new IntSet2());
    }
  }

  /**
   * Here the delegate is an {@link IntSet3}.
   */
  public static final class Test4Using3 extends InstrumentedIntSetTest {
    protected InstrumentedIntSet factory() {
      return new InstrumentedIntSet4(new IntSet3());
    }
  }

  /**
   * Here the delegate is an {@link IntSet4}.
   */
  public static final class Test4Using4 extends InstrumentedIntSetTest {
    protected InstrumentedIntSet factory() {
      return new InstrumentedIntSet4(new IntSet4());
    }
  }
  public static final class Test4Using4Using4 extends InstrumentedIntSetTest {
    protected InstrumentedIntSet factory() {
      return new InstrumentedIntSet4(new InstrumentedIntSet4(new IntSet4()));
    }
  }
}
