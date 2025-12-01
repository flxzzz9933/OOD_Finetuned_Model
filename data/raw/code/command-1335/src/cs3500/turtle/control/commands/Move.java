package cs3500.turtle.control.commands;

import cs3500.turtle.control.TracingTurtleCommand;
import cs3500.turtle.control.UndoableTTCmd;
import cs3500.turtle.tracingmodel.TracingTurtleModel;

/**
 * Created by blerner on 10/10/16.
 */
public class Move implements UndoableTTCmd {
  double d;

  public Move(Double d) {
    this.d = d;
  }

  @Override
  public void execute(TracingTurtleModel model) {
    model.move(this.d);
  }

  @Override
  public void undo(TracingTurtleModel m) {
    m.move(-this.d);
  }
}
