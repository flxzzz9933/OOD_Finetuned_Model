package cs3500.adapters;

import java.util.Iterator;

import cs3500.lec09.IntSet1;
import cs3500.lec09.IntSet2;
import cs3500.lec09.IntSet2Impl;

public class IntSet2ToIntSet1 implements IntSet1 {

  private IntSet2 delegate;

  public IntSet2ToIntSet1(IntSet2 other) {
    this.delegate = other;
  }

  //IntSet2Impl.empty()
  //IntSet2Impl.singleton(int)


  @Override
  public void add(int value) {
    delegate.unionWith(IntSet2Impl.singleton(value));
  }

  @Override
  public void remove(int value) {
    delegate.differenceFrom(IntSet2Impl.singleton(value));
  }

  @Override
  public boolean member(int value) {
    return delegate.isSupersetOf(IntSet2Impl.singleton(value));
  }

  @Override
  public Iterator<Integer> iterator() {
    return delegate.asList().iterator();
  }
}
