package cs3500.turtle.control.commands;

import cs3500.turtle.control.TracingTurtleCommand;
import cs3500.turtle.tracingmodel.TracingTurtleModel;

/**
 * Created by blerner on 10/10/16.
 */
public class Square implements TracingTurtleCommand {
  double d;

  public Square(Double d) {
    this.d = d;
  }

  @Override
  public void execute(TracingTurtleModel model) {
    for (int i = 0; i < 4; i++) {
      model.trace(this.d);
      model.turn(90);
    }
  }
}
