package cs3500.turtle.control;

import cs3500.turtle.control.commands.Koch;
import cs3500.turtle.control.commands.Move;
import cs3500.turtle.control.commands.Retrieve;
import cs3500.turtle.control.commands.Save;
import cs3500.turtle.control.commands.Square;
import cs3500.turtle.control.commands.Trace;
import cs3500.turtle.control.commands.Turn;
import cs3500.turtle.tracingmodel.TracingTurtleModel;
import cs3500.turtle.view.IView;
import cs3500.turtle.view.ViewActions;

import java.util.Scanner;

/**
 * This is a controller that is very similar to the
 * CommandController class. The main difference
 * is that the main is replaced with the controller
 * method processCommand
 */
public class MVCCommandController implements IController, ViewActions {
  private TracingTurtleModel model;
  private IView view;

  public MVCCommandController(TracingTurtleModel model, IView view) {
    this.model = model;
    this.view = view;
  }

  @Override
  public void startProgram() {
    //TODO: As the Observer, subscribe to the Subject
    view.setViewActions(this);
    view.makeVisible();
  }

  @Override
  public String processCommand(String command) {
    StringBuilder output = new StringBuilder();
    Scanner s = new Scanner(command);
    TracingTurtleCommand cmd = null;


    while (s.hasNext()) {
      String in = s.next();

      switch (in) {
        case "move":
          cmd = new Move(s.nextDouble());
          break;
        case "trace":
          cmd = new Trace(s.nextDouble());
          break;
        case "turn":
          cmd = new Turn(s.nextDouble());
          break;
        case "square":
          cmd = new Square(s.nextDouble());
          break;
        case "koch":
          cmd = new Koch(s.nextDouble(), s.nextInt());
          break;
        case "save":
          cmd = new Save();
          break;
        case "retrieve":
          cmd = new Retrieve();
          break;
        default:
          output.append(String.format("Unknown command %s", in));
          cmd = null;
          break;
      }
      if (cmd != null) {
        cmd.execute(model);
        output.append("Successfully executed: " + command);
      }
    }


    return output.toString();
  }

  @Override
  public void executeCommand(String command) {
    processCommand(command);
    view.refresh();
  }

  @Override
  public void exitProgram() {
    System.exit(0);
  }
}
