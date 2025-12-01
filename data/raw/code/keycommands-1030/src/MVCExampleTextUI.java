import view.EchoFrame;
import view.GUIView;

public class MVCExampleTextUI {
    public static void main(String []args) {
        IModel model = new Model();
        GUIView view = new EchoFrame(); //This view no longer needs the model.
        GUIController controller = new GUIController(model, view); //constructor sets up everything
        //alternatively, expose a method that sets the ViewActions and sets the view to be visible

        /*
         * // OLD WAY!
        IModel model = new Model();
        IView view = new TextView(System.out);
        IController controller = new TextController(model,System.in,view);
        controller.runProgram();
         */
    }
}
