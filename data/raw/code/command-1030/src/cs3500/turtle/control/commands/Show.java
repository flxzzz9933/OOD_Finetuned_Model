package cs3500.turtle.control.commands;

import cs3500.turtle.control.TracingTurtleCommand;
import cs3500.turtle.tracingmodel.Line;
import cs3500.turtle.tracingmodel.TracingTurtleModel;

public class Show implements TracingTurtleCommand {
  @Override
  public void execute(TracingTurtleModel model) {
    for (Line l : model.getLines()) {
      System.out.println(l);
    }
  }
}
