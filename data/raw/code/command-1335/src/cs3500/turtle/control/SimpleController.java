package cs3500.turtle.control;

import java.util.InputMismatchException;
import java.util.List;
import java.util.Scanner;

import cs3500.turtle.tracingmodel.Line;
import cs3500.turtle.tracingmodel.SmarterTurtle;
import cs3500.turtle.tracingmodel.TracingTurtleModel;

/**
 * Created by blerner on 10/10/16.
 */
public class SimpleController {
  public static void main(String[] args) {
    Scanner s = new Scanner(System.in);
    TracingTurtleModel m = new SmarterTurtle();
    while (s.hasNext()) {
      String in = s.next();
      switch(in) {
        case "q":
        case "quit":
          return;
        case "show":
          showHelper(m);
          break;
        case "move":
          try {
            double d = s.nextDouble();
            m.move(d);
          } catch (InputMismatchException ime) {
            System.out.println("Bad length to move");
          }
          break;
        case "trace":
          try {
            double d = s.nextDouble();
            m.trace(d);
          } catch (InputMismatchException ime) {
            System.out.println("Bad length to trace");
          }
          break;
        case "turn":
          try {
            double d = s.nextDouble();
            turnHelper(m, d);
          } catch (InputMismatchException ime) {
            System.out.println("Bad length to turn");
          }
          break;
        case "square":
          try {
            double d = s.nextDouble();
            squareHelper(m, d);
          } catch (InputMismatchException ime) {
            System.out.println("Bad length to turn");
          }
          break;
        default:
          System.out.println(String.format("Unknown command %s", in));
          break;
      }
    }
  }



  interface Command {
    void execute(TracingTurtleModel model);
  }

  class ShowLines implements Command {

    @Override
    public void execute(TracingTurtleModel model) {
      for (Line l : model.getLines()) {
        System.out.println(l);
      }
    }
  }

  private static void showHelper(TracingTurtleModel m) {
    for (Line l : m.getLines()) {
      System.out.println(l);
    }
  }

  class Turn implements Command {
    private double angle;

    public Turn(double angle) {
      this.angle = angle;
    }

    @Override
    public void execute(TracingTurtleModel model) {
      model.turn(angle);
    }
  }

  private static void turnHelper(TracingTurtleModel m, double d) {
    m.turn(d);
  }

  private static void squareHelper(TracingTurtleModel m, double d) {
    m.trace(d);
    m.turn(90);
    m.trace(d);
    m.turn(90);
    m.trace(d);
    m.turn(90);
    m.trace(d);
    m.turn(90);
  }
}
