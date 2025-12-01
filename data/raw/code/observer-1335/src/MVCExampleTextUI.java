import model.IModel;
import model.Model;
import view.EchoFrame;
import view.IGUIView;

public class MVCExampleTextUI {
    public static void main(String []args) {
        IModel model = new Model();
        IGUIView view = new EchoFrame(model);
        GUIController controller = new GUIController(model, view);
        //So... when do we make the view visible?

        /*
        IModel model = new Model();
        IView view = new TextView(System.out);
        IController controller = new TextController(model,System.in,view);
        controller.runProgram();
         */
    }
}
