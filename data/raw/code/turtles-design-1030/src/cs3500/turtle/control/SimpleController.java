package cs3500.turtle.control;

import java.util.List;
import java.util.Scanner;

import cs3500.turtle.tracingmodel.SmarterTurtle;
import cs3500.turtle.tracingmodel.TracingTurtleModel;

/**
 * Created by blerner on 10/10/16. Edited by lnunez on 02/05/24.
 */
public class SimpleController {
  public static void main(String[] args) {
    Scanner scan = new Scanner(System.in);
    TracingTurtleModel model = new SmarterTurtle();
    //TODO: Add the controller into this method
    while(scan.hasNext()) {
      //1. Read a command
      //String cmd = scan.next();
      //2. Process the command

      switch(scan.next()) { //Can make it lowercase
        case "Q":
        case "q":
          return;
        case "move":
          model.move(scan.nextDouble());
          break;
        case "trace":
          model.trace(scan.nextDouble());
          break;
        case "square":
          double length = scan.nextDouble();
          model.trace(length);
          model.turn(90);
          model.trace(length);
          model.turn(90);
          model.trace(length);
          model.turn(90);
          model.trace(length);
          model.turn(90);
      }
    }
  }
}
