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
    Scanner s = new Scanner(System.in);
    TracingTurtleModel m = new SmarterTurtle();
    //TODO: Add the controller into this method

    while(s.hasNext()) {
      String cmd = s.next();
      switch(cmd) {
        case "move":
          m.move(s.nextDouble());
          break;
        default:
          System.out.println("Invalid command. Sad");
      }
    }
  }
}
