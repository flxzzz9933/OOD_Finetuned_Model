package cs3500.adapters;

import java.awt.*;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import cs3500.lec09.IntSet1Impl;
import cs3500.lec09.IntSet2;

public class IntSet1ToIntSet2 extends IntSet1Impl implements IntSet2 {
  @Override
  public void unionWith(IntSet2 other) {
    //loop over other
    for(int val : other.asList()) {
      //if not a member of us, add them
      super.add(val);
    }
  }

  @Override
  public void differenceFrom(IntSet2 other) {
    for(int val : other.asList()) {
      super.remove(val);
    }
  }

  @Override
  public boolean isSupersetOf(IntSet2 other) {
    for(int val : other.asList()) {
      if(!super.member(val)) {
        return false;
      }
    }
    return true;
  }

  @Override
  public List<Integer> asList() {
    Iterator<Integer> it = super.iterator();
    List<Integer> ans = new ArrayList<>();
    /*
    while(it.hasNext()) {
      ans.add(it.next());
    }
    */
    for(int val : this) {
      ans.add(val);
    }

    return ans;
  }
}
